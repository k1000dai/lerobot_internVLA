# Copyright 2025 The HuggingFace Inc. team. All rights reserved.

import copy
import contextlib
import math
from typing import Any

import torch
from torch import nn

try:
    from transformers import AutoConfig, AutoModel, AutoProcessor, InternVLForConditionalGeneration
except Exception:  # pragma: no cover - Import guarded for environments without transformers>=4.55
    AutoConfig = None
    AutoModel = None
    AutoProcessor = None
    InternVLForConditionalGeneration = None


def apply_rope(x: torch.Tensor, positions: torch.Tensor, max_wavelength: int = 10_000) -> torch.Tensor:
    """Apply RoPE positions [B,L] to x [B,L,H,Dh]. Generic variant; matches shapes used in attention below."""
    d_half = x.shape[-1] // 2
    device = x.device
    dtype = x.dtype
    x = x.to(torch.float32)

    freq_exponents = (2.0 / x.shape[-1]) * torch.arange(d_half, dtype=torch.float32, device=device)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None].to(torch.float32) / timescale[None, None, :]
    radians = radians[..., None, :]
    sin, cos = torch.sin(radians), torch.cos(radians)

    x1, x2 = x.split(d_half, dim=-1)
    out = torch.empty_like(x)
    out[..., :d_half] = x1 * cos - x2 * sin
    out[..., d_half:] = x2 * cos + x1 * sin
    return out.to(dtype)


def get_intermediate_size(hidden_dim: int, ffn_dim_multiplier: float = 4.0, multiple_of: int = 256) -> int:
    hidden_dim = int(2 * hidden_dim / 3)
    hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim


class InternVLWithExpertModel(nn.Module):
    """
    A lightweight layer-by-layer attention mixer between InternVL text transformer and a smaller expert transformer
    used for action prediction. Design mirrors SmolVLMWithExpertModel and Pi0's PaliGemmaWithExpert approach.
    """

    def __init__(
        self,
        model_id: str = "OpenGVLab/InternVL3_5-4B-HF",
        *,
        load_vlm_weights: bool = True,
        train_expert_only: bool = True,
        freeze_vision_encoder: bool = True,
        attention_mode: str = "cross_attn",
        num_expert_layers: int = -1,
        num_vlm_layers: int = -1,
        self_attn_every_n_layers: int = 2,
        expert_width_multiplier: float = 0.75,
        knowledge_insulation: bool = False,
    ) -> None:
        super().__init__()
        if InternVLForConditionalGeneration is None:
            raise ImportError(
                "transformers>=4.55.0 is required for InternVL. Please upgrade transformers to use InternVLA."
            )

        self.train_expert_only = train_expert_only
        self.freeze_vision_encoder = freeze_vision_encoder
        self.attention_mode = attention_mode
        self.self_attn_every_n_layers = self_attn_every_n_layers
        self.knowledge_insulation = knowledge_insulation

        # Load InternVL VLM
        if load_vlm_weights:
            self.vlm = InternVLForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="auto"
            )
            config = self.vlm.config
        else:
            config = AutoConfig.from_pretrained(model_id)
            self.vlm = InternVLForConditionalGeneration(config)

        # Processor (for external tokenization or image preproc if needed)
        try:
            self.processor = AutoProcessor.from_pretrained(model_id)
        except Exception:
            self.processor = None

        # Truncate VLM depth if requested (Qwen family layout differs; be robust)
        lm = self.get_vlm_model().language_model

        def _resolve_layers_owner(text_model):
            # Prefer the direct container (e.g., Qwen3Model has .layers)
            layers = getattr(text_model, "layers", None)
            if layers is not None:
                return text_model, layers
            # Fallback to common inner modules
            for attr in ("model", "text_model", "transformer", "decoder"):
                owner = getattr(text_model, attr, None)
                if owner is None:
                    continue
                layers = getattr(owner, "layers", None)
                if layers is not None:
                    return owner, layers
            raise AttributeError("Could not find a 'layers' container in InternVL language_model.")

        layers_owner, layers_list = _resolve_layers_owner(lm)
        if num_vlm_layers is not None and num_vlm_layers > 0:
            from torch import nn as _nn

            new_layers = _nn.ModuleList(list(layers_list)[:num_vlm_layers])
            setattr(layers_owner, "layers", new_layers)
            layers_list = new_layers
        self.num_vlm_layers = len(layers_list)

        # Build expert (Qwen-like) from text_config scaled down
        text_cfg = copy.deepcopy(config.text_config)
        hidden_size = text_cfg.hidden_size
        text_cfg.hidden_size = int(hidden_size * expert_width_multiplier)
        text_cfg.intermediate_size = get_intermediate_size(text_cfg.hidden_size)
        text_cfg.num_hidden_layers = self.num_vlm_layers
        if num_expert_layers is not None and num_expert_layers > 0:
            assert self.num_vlm_layers % num_expert_layers == 0, (
                f"VLM layers ({self.num_vlm_layers}) must be a multiple of num_expert_layers ({num_expert_layers})."
            )
            text_cfg.num_hidden_layers = num_expert_layers
        self.lm_expert = AutoModel.from_config(text_cfg)

        # Cache config-driven dims for attention mixing
        self.num_attention_heads = config.text_config.num_attention_heads
        self.num_key_value_heads = config.text_config.num_key_value_heads
        self.head_dim = config.text_config.head_dim
        self.vlm_hidden_size = config.text_config.hidden_size
        self.expert_hidden_size = text_cfg.hidden_size

        # If using cross-attention mixing, adapt expert K/V projections to accept VLM memory width
        if "cross" in self.attention_mode:
            try:
                expert_layers = getattr(self.lm_expert, "layers", None)
                if expert_layers is None and hasattr(self.lm_expert, "model"):
                    expert_layers = getattr(self.lm_expert.model, "layers", None)
                if expert_layers is None:
                    raise AttributeError
                in_features = config.text_config.num_key_value_heads * config.text_config.head_dim
                out_features = text_cfg.num_key_value_heads * text_cfg.head_dim
                bias_flag = getattr(text_cfg, "attention_bias", False)
                for layer_idx in range(len(expert_layers)):
                    if self.self_attn_every_n_layers > 0 and layer_idx % self.self_attn_every_n_layers == 0:
                        # Keep self-attn layers untouched when interleaving
                        continue
                    layer = expert_layers[layer_idx]
                    layer.self_attn.k_proj = nn.Linear(in_features, out_features, bias=bias_flag)
                    layer.self_attn.v_proj = nn.Linear(in_features, out_features, bias=bias_flag)
            except Exception:
                # Fallback: leave expert projections unchanged for self-attn only mode
                pass

        self._set_requires_grad()

    def _set_requires_grad(self) -> None:
        if self.freeze_vision_encoder:
            self.get_vlm_model().vision_tower.eval()
            for p in self.get_vlm_model().vision_tower.parameters():
                p.requires_grad = False
        if self.train_expert_only:
            self.vlm.eval()
            for p in self.vlm.parameters():
                p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_vision_encoder:
            self.get_vlm_model().vision_tower.eval()
        if self.train_expert_only:
            self.vlm.eval()

    # Accessors
    def get_vlm_model(self):
        return self.vlm.model

    # Embedding helpers
    def embed_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values expected as (B,3,H,W), float/bfloat; InternVL handles projection
        image_hidden_states = self.vlm.get_image_features(pixel_values=pixel_values)
        return image_hidden_states  # (B, N_img, D_txt)

    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.vlm.get_input_embeddings()(tokens)

    # Attention primitive
    def get_attention_interface(self):
        return self._eager_attention_forward

    def _eager_attention_forward(
        self,
        attention_mask: torch.Tensor,
        batch_size: int,
        head_dim: int,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> torch.Tensor:
        num_att_heads = self.num_attention_heads
        num_key_value_heads = self.num_key_value_heads
        num_key_value_groups = num_att_heads // num_key_value_heads

        seq_len = key_states.shape[1]
        key_states = key_states[:, :, :, None, :].expand(batch_size, seq_len, num_key_value_heads, num_key_value_groups, head_dim)
        key_states = key_states.reshape(batch_size, seq_len, num_key_value_heads * num_key_value_groups, head_dim)
        value_states = value_states[:, :, :, None, :].expand(batch_size, seq_len, num_key_value_heads, num_key_value_groups, head_dim)
        value_states = value_states.reshape(batch_size, seq_len, num_key_value_heads * num_key_value_groups, head_dim)

        q = query_states.to(dtype=torch.float32).transpose(1, 2)
        k = key_states.to(dtype=torch.float32).transpose(1, 2)
        att = (q @ k.transpose(2, 3)) * (head_dim ** -0.5)
        big_neg = torch.finfo(att.dtype).min
        att = torch.where(attention_mask[:, None, :, :], att, big_neg)
        probs = torch.softmax(att, dim=-1).to(dtype=value_states.dtype)
        out = probs @ value_states.permute(0, 2, 1, 3)
        out = out.permute(0, 2, 1, 3).reshape(batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim)
        return out

    # One transformer block across VLM + Expert
    def _forward_attn_layer(
        self,
        model_layers: list[list[Any] | Any],
        inputs_embeds: list[torch.Tensor | None],
        layer_idx: int,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        batch_size: int,
        head_dim: int,
        *,
        use_cache: bool,
        fill_kv_cache: bool,
        past_key_values: dict | None,
        adarms_cond: torch.Tensor | None,
    ):
        # Gather q,k,v for present inputs
        qs, ks, vs = [], [], []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is None:
                continue
            layer = model_layers[i][layer_idx]
            # For KI, insulate VLM stream (i==0) by computing q/k/v without grad
            no_grad_ctx = torch.no_grad() if (self.knowledge_insulation and i == 0) else contextlib.nullcontext()
            with no_grad_ctx:
                hidden_states = layer.input_layernorm(hidden_states)
                # Inject knowledge-insulation condition into expert stream (simple FiLM add)
                if adarms_cond is not None and i == 1:
                    hidden_states = hidden_states + adarms_cond[:, None, :].to(hidden_states.dtype)

                shape = (*hidden_states.shape[:-1], -1, layer.self_attn.head_dim)
                hs = hidden_states.to(dtype=layer.self_attn.q_proj.weight.dtype)
                q = layer.self_attn.q_proj(hs).view(shape)
                k = layer.self_attn.k_proj(hs).view(shape)
                v = layer.self_attn.v_proj(hs).view(shape)
            qs.append(q)
            ks.append(k)
            vs.append(v)

        q = torch.cat(qs, dim=1)
        k = torch.cat(ks, dim=1)
        v = torch.cat(vs, dim=1)
        seq_len = q.shape[1]
        pos = position_ids[:, :seq_len]
        att_mask = attention_mask[:, :seq_len, :seq_len]
        q = apply_rope(q, pos)
        k = apply_rope(k, pos)

        # KV cache handling
        if use_cache and past_key_values is None:
            past_key_values = {}
        if use_cache:
            if fill_kv_cache:
                past_key_values[layer_idx] = {"key_states": k, "value_states": v}
            else:
                k = torch.cat([past_key_values[layer_idx]["key_states"], k], dim=1)
                v = torch.cat([past_key_values[layer_idx]["value_states"], v], dim=1)

        att_out = self.get_attention_interface()(att_mask, batch_size, head_dim, q, k, v)
        return [att_out], past_key_values

    def _forward_cross_attn_layer(
        self,
        model_layers: list[list[Any] | Any],
        inputs_embeds: list[torch.Tensor | None],
        layer_idx: int,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        batch_size: int,
        head_dim: int,
        *,
        use_cache: bool,
        fill_kv_cache: bool,
        past_key_values: dict | None,
        adarms_cond: torch.Tensor | None,
    ):
        att_outputs: list[torch.Tensor] = []
        # Prefix pass through VLM stream to generate KV
        if len(inputs_embeds) == 2 and past_key_values is None:
            seq_len = inputs_embeds[0].shape[1]
            pos_prefix, pos_expert = position_ids[:, :seq_len], position_ids[:, seq_len:]
            mask_prefix = attention_mask[:, :seq_len, :seq_len]
            vlm_layer = model_layers[0][layer_idx]
            # For KI, compute VLM prefix q/k/v without grad
            no_grad_ctx = torch.no_grad() if self.knowledge_insulation else contextlib.nullcontext()
            with no_grad_ctx:
                hs = vlm_layer.input_layernorm(inputs_embeds[0])
                q_shape = (*hs.shape[:-1], -1, vlm_layer.self_attn.head_dim)
                hs = hs.to(dtype=vlm_layer.self_attn.q_proj.weight.dtype)
                q = vlm_layer.self_attn.q_proj(hs).view(q_shape)
                k = vlm_layer.self_attn.k_proj(hs).view(q_shape)
                v = vlm_layer.self_attn.v_proj(hs).view(q_shape)
                q = apply_rope(q, pos_prefix)
                k = apply_rope(k, pos_prefix)
            att_out = self.get_attention_interface()(mask_prefix, batch_size, head_dim, q, k, v)
            att_outputs.append(att_out)
        else:
            pos_expert = position_ids

        # Cache handling for KV
        if use_cache and past_key_values is None:
            past_key_values = {}
        if use_cache:
            if fill_kv_cache:
                past_key_values[layer_idx] = {"key_states": k, "value_states": v}
            else:
                k = past_key_values[layer_idx]["key_states"]
                v = past_key_values[layer_idx]["value_states"]

        # Expert queries attend to VLM K,V (projected into expert width)
        expert_layer = model_layers[1][layer_idx]
        if expert_layer is not None:
            hs = expert_layer.input_layernorm(inputs_embeds[1])
            if adarms_cond is not None:
                hs = hs + adarms_cond[:, None, :].to(hs.dtype)
            exp_shape = (*hs.shape[:-1], -1, expert_layer.self_attn.head_dim)
            hs = hs.to(dtype=expert_layer.self_attn.q_proj.weight.dtype)
            q_e = expert_layer.self_attn.q_proj(hs).view(exp_shape)

            # Project VLM K,V into expert head dims
            k_flat = k.to(dtype=expert_layer.self_attn.k_proj.weight.dtype).view(*k.shape[:2], -1)
            v_flat = v.to(dtype=expert_layer.self_attn.v_proj.weight.dtype).view(*v.shape[:2], -1)
            k_e = expert_layer.self_attn.k_proj(k_flat).view(*k_flat.shape[:-1], -1, expert_layer.self_attn.head_dim)
            v_e = expert_layer.self_attn.v_proj(v_flat).view(*v_flat.shape[:-1], -1, expert_layer.self_attn.head_dim)

            # Reset expert positions relative to expert tokens
            pos_expert = pos_expert - torch.min(pos_expert, dim=1, keepdim=True).values
            # Apply RoPE to expert queries using expert token positions
            q_e = apply_rope(q_e, pos_expert)
            # Do NOT re-apply RoPE to keys derived from VLM prefix; they were already rotated with pos_prefix

            # Expert attends to VLM memory
            att_mask = attention_mask[:, -q_e.shape[1] :, : k_e.shape[1]]
            att_out = self.get_attention_interface()(att_mask, batch_size, expert_layer.self_attn.head_dim, q_e, k_e, v_e)
            att_outputs.append(att_out)

        return att_outputs, past_key_values

    def forward(
        self,
        *,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        past_key_values: dict | None,
        inputs_embeds: list[torch.Tensor | None],  # [prefix_embs, suffix_embs]
        use_cache: bool,
        fill_kv_cache: bool,
        adarms_cond: torch.Tensor | None = None,
    ):
        # Resolve text/expert layer lists robustly
        lm = self.get_vlm_model().language_model

        def _resolve_layers(module):
            # Prefer direct .layers (Qwen3Model)
            layers = getattr(module, "layers", None)
            if layers is not None:
                return module, layers
            # Else try common wrappers
            for attr in ("model", "text_model", "transformer", "decoder"):
                base = getattr(module, attr, None)
                if base is None:
                    continue
                layers = getattr(base, "layers", None)
                if layers is not None:
                    return base, layers
            raise AttributeError("Could not resolve language model layers for InternVL.")

        text_base, text_layers = _resolve_layers(lm)
        expert_layers = getattr(self.lm_expert, "layers", None)
        if expert_layers is None and hasattr(self.lm_expert, "model"):
            expert_layers = getattr(self.lm_expert.model, "layers", None)
        if expert_layers is None:
            raise AttributeError("Expert model does not expose a 'layers' attribute.")

        model_layers = [text_layers, expert_layers]

        # Discover batch/shape
        batch_size = next(e for e in inputs_embeds if e is not None).shape[0]
        head_dim = self.head_dim
        nlayers = len(model_layers[0])

        for layer_idx in range(nlayers):
            # Choose self or cross attention mixing
            if (
                fill_kv_cache
                or "cross" not in self.attention_mode
                or (self.self_attn_every_n_layers > 0 and layer_idx % self.self_attn_every_n_layers == 0)
            ):
                att_outs, past_key_values = self._forward_attn_layer(
                    model_layers,
                    inputs_embeds,
                    layer_idx,
                    position_ids,
                    attention_mask,
                    batch_size,
                    head_dim,
                    use_cache=use_cache,
                    fill_kv_cache=fill_kv_cache,
                    past_key_values=past_key_values,
                    adarms_cond=adarms_cond,
                )
            else:
                att_outs, past_key_values = self._forward_cross_attn_layer(
                    model_layers,
                    inputs_embeds,
                    layer_idx,
                    position_ids,
                    attention_mask,
                    batch_size,
                    head_dim,
                    use_cache=use_cache,
                    fill_kv_cache=fill_kv_cache,
                    past_key_values=past_key_values,
                    adarms_cond=adarms_cond,
                )

            # MLP + residual
            outputs_embeds: list[torch.Tensor | None] = []
            start = 0
            for i, hidden_states in enumerate(inputs_embeds):
                layer = model_layers[i][layer_idx]
                att_out = att_outs[i] if i < len(att_outs) else att_outs[0]
                if hidden_states is None:
                    outputs_embeds.append(None)
                    continue
                end = start + hidden_states.shape[1]
                if att_out.dtype != layer.self_attn.o_proj.weight.dtype:
                    att_out = att_out.to(layer.self_attn.o_proj.weight.dtype)
                att_slice = att_out[:, start:end]
                out = layer.self_attn.o_proj(att_slice)
                # Ensure same dtype for residual add
                if hidden_states.dtype != out.dtype:
                    hidden_states = hidden_states.to(dtype=out.dtype)
                out = out + hidden_states
                x = layer.post_attention_layernorm(out)
                # Simple FiLM gating on expert after post-attn LN
                if adarms_cond is not None and i == 1:
                    x = x + adarms_cond[:, None, :].to(x.dtype)
                # Match MLP weights dtype
                mlp_dtype = layer.mlp.gate_proj.weight.dtype if hasattr(layer.mlp, "gate_proj") else x.dtype
                if x.dtype != mlp_dtype:
                    x = x.to(dtype=mlp_dtype)
                x = layer.mlp(x)
                x = x + out
                outputs_embeds.append(x)
                start = end if len(att_outs) == 1 else 0
            inputs_embeds = outputs_embeds

        # Final norms on each stream
        def _get_norm(m):
            for name in ("norm", "final_layernorm", "ln_f", "layer_norm"):
                n = getattr(m, name, None)
                if n is not None:
                    return n
            return None

        text_norm_owner = text_base if _get_norm(text_base) is not None else lm
        expert_norm_owner = self.lm_expert if _get_norm(self.lm_expert) is not None else getattr(self.lm_expert, "model", self.lm_expert)

        outputs_normed: list[torch.Tensor | None] = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is None:
                outputs_normed.append(None)
                continue
            norm = _get_norm(text_norm_owner) if i == 0 else _get_norm(expert_norm_owner)
            outputs_normed.append(norm(hidden_states) if norm is not None else hidden_states)

        return outputs_normed, past_key_values
