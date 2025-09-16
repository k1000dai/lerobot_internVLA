#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of the VLA-Adapter policy.

The code mirrors the architecture presented in the paper "VLA-Adapter: An Effective
Paradigm for Tiny-Scale Vision-Language-Action Model".  The policy is made of two
main pieces:

1. A lightweight vision-language backbone that returns per-layer hidden states as well as
   the latent corresponding to learnable action queries.
2. A policy network composed of `BridgeAttention` blocks which inject the multi-layer
   conditions into the action latent space.

This module focuses on the Bridge Attention logic.  The backbone is intentionally kept
simple – a small Transformer over visual and textual tokens – so that the implementation
remains self-contained while exercising the same data flow as the original method.
"""

from __future__ import annotations

from collections import deque
from typing import Iterable

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer

from lerobot.constants import ACTION, OBS_STATE
from lerobot.policies.normalize import Normalize, Unnormalize
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import populate_queues

from .configuration_vla_adapter import VLAAdapterConfig


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def _resize_with_pad(img: Tensor, width: int, height: int, pad_value: float = 0.0) -> Tensor:
    """Resize an image tensor to (height, width) while keeping the aspect ratio."""

    if img.ndim != 4:
        raise ValueError(f"(batch, channels, height, width) expected, got {img.shape}")

    cur_h, cur_w = img.shape[2:]
    ratio = max(cur_w / width, cur_h / height)
    new_h, new_w = int(cur_h / ratio), int(cur_w / ratio)
    resized = F.interpolate(img, size=(new_h, new_w), mode="bilinear", align_corners=False)
    pad_h = max(0, height - new_h)
    pad_w = max(0, width - new_w)
    padded = F.pad(resized, (pad_w, 0, pad_h, 0), value=pad_value)
    return padded


def _pad_vector(vector: Tensor, new_dim: int) -> Tensor:
    """Pad a tensor on the last dimension with zeros to reach ``new_dim``."""

    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector


def _invert_pad_mask(mask: Tensor | None) -> Tensor | None:
    if mask is None:
        return None
    return ~mask


# -----------------------------------------------------------------------------
# Tiny multimodal backbone
# -----------------------------------------------------------------------------


class PatchEmbed(nn.Module):
    """Simple patch embedding with a convolution."""

    def __init__(self, in_channels: int, hidden_size: int, patch_size: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected (batch, channels, height, width), got {x.shape}")
        out = self.proj(x)
        out = out.flatten(2).transpose(1, 2)
        return out


class QwenBackbone(nn.Module):
    """Wrapper around Qwen2.5-0.5B that exposes layer-wise states and action queries."""

    def __init__(self, config: VLAAdapterConfig) -> None:
        super().__init__()
        try:
            self.model = AutoModel.from_pretrained(
                config.vlm_model_name,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                output_hidden_states=True,
            )
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Failed to load the Qwen backbone. Ensure transformers is installed and the model is available."
            ) from exc

        if config.freeze_vlm:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

        self.hidden_size = getattr(self.model.config, "hidden_size", None)
        if self.hidden_size is None:
            raise AttributeError("Loaded Qwen model does not expose 'hidden_size'.")

        self.num_layers = getattr(self.model.config, "num_hidden_layers", None)
        if self.num_layers is None:
            raise AttributeError("Loaded Qwen model does not expose 'num_hidden_layers'.")

        self.num_attention_heads = getattr(self.model.config, "num_attention_heads", 8)

        self.num_action_queries = config.num_action_queries
        self.patch_embed = PatchEmbed(3, self.hidden_size, config.patch_size)
        self.action_queries = nn.Parameter(torch.zeros(self.num_action_queries, self.hidden_size))
        nn.init.trunc_normal_(self.action_queries, std=0.02)

    @property
    def device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:  # pragma: no cover
            return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        try:
            return next(self.model.parameters()).dtype
        except StopIteration:  # pragma: no cover
            return torch.float32

    def forward(
        self,
        images: Iterable[Tensor],
        lang_tokens: Tensor,
        lang_attention_mask: Tensor,
    ) -> dict:
        device = self.device
        dtype = self.dtype

        batch_size = lang_tokens.shape[0]
        seqs: list[Tensor] = []
        masks: list[Tensor] = []

        for img in images:
            if img.ndim == 5:
                img = img[:, -1]
            img = img.to(device=device, dtype=dtype)
            vis_tokens = self.patch_embed(img)
            seqs.append(vis_tokens)
            masks.append(torch.ones(batch_size, vis_tokens.shape[1], dtype=torch.bool, device=device))

        lang_tokens = lang_tokens.to(device=device)
        lang_emb = self.model.get_input_embeddings()(lang_tokens).to(dtype=dtype)
        lang_mask = lang_attention_mask.to(device=device, dtype=torch.bool)
        seqs.append(lang_emb)
        masks.append(lang_mask)

        aq = self.action_queries.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
        seqs.append(aq)
        masks.append(torch.ones(batch_size, aq.shape[1], dtype=torch.bool, device=device))

        seq = torch.cat(seqs, dim=1)
        mask = torch.cat(masks, dim=1)

        position_ids = torch.arange(seq.shape[1], device=device).unsqueeze(0).expand(batch_size, -1)
        outputs = self.model(
            inputs_embeds=seq,
            attention_mask=mask.to(dtype=torch.long),
            position_ids=position_ids,
            output_hidden_states=True,
            use_cache=False,
        )

        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Backbone did not return hidden states. Ensure output_hidden_states=True is supported.")

        # Skip embedding layer (index 0)
        hidden_states = hidden_states[1:]
        raw_features = []
        aq_features = []
        for hs in hidden_states:
            hs = hs.to(dtype=torch.float32)
            raw_features.append(hs[:, :-self.num_action_queries, :])
            aq_features.append(hs[:, -self.num_action_queries :, :])

        raw_mask = mask[:, :-self.num_action_queries]
        aq_mask = mask[:, -self.num_action_queries :]

        return {
            "raw_features": raw_features,
            "action_query_features": aq_features,
            "raw_attention_mask": raw_mask,
            "action_query_mask": aq_mask,
        }


# -----------------------------------------------------------------------------
# Bridge Attention
# -----------------------------------------------------------------------------


class BridgeAttentionBlock(nn.Module):
    """Bridge Attention module from the VLA-Adapter paper."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        ratio_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_raw = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.cross_aq = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

        self.q_norm = nn.LayerNorm(hidden_size)
        self.raw_norm = nn.LayerNorm(hidden_size)
        self.aq_norm = nn.LayerNorm(hidden_size)

        self.concat_norm = nn.LayerNorm(hidden_size * 3)
        self.out_proj = nn.Linear(hidden_size * 3, hidden_size)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size),
        )

        self.g = nn.Parameter(torch.tensor(ratio_init))

    def forward(
        self,
        action_latent: Tensor,
        raw_cond: Tensor,
        aq_cond: Tensor,
        proprio_token: Tensor | None = None,
        raw_mask: Tensor | None = None,
        aq_mask: Tensor | None = None,
    ) -> Tensor:
        q = self.q_norm(action_latent)
        raw = self.raw_norm(raw_cond)
        aq = self.aq_norm(aq_cond)

        if proprio_token is not None:
            proprio_token = proprio_token.unsqueeze(1)
            if aq_mask is not None:
                proprio_mask = torch.ones(aq_mask.shape[0], 1, dtype=aq_mask.dtype, device=aq_mask.device)
                aq_mask = torch.cat([aq_mask, proprio_mask], dim=1)
            aq = torch.cat([aq, proprio_token.expand(aq.shape[0], 1, -1)], dim=1)

        cross_raw, _ = self.cross_raw(q, raw, raw, key_padding_mask=_invert_pad_mask(raw_mask))
        cross_aq, _ = self.cross_aq(q, aq, aq, key_padding_mask=_invert_pad_mask(aq_mask))
        self_out, _ = self.self_attn(q, q, q)

        raw_weight = torch.tanh(self.g)
        concat = torch.cat([cross_raw * raw_weight, cross_aq, self_out], dim=-1)
        bridge = self.out_proj(self.concat_norm(concat))

        x = action_latent + bridge
        x = x + self.ffn(self.ffn_norm(x))
        return x


class BridgePolicy(nn.Module):
    """Policy head composed of stacked Bridge Attention blocks."""

    def __init__(self, config: VLAAdapterConfig, hidden_size: int, num_layers: int, num_heads: int) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.initial_embed = nn.Sequential(
            nn.LayerNorm(config.max_action_dim),
            nn.Linear(config.max_action_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.proprio_proj = nn.Linear(config.max_state_dim, hidden_size)
        self.layers = nn.ModuleList(
            [
                BridgeAttentionBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=config.policy_mlp_ratio,
                    ratio_init=config.bridge_ratio_init,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_size)
        self.action_head = nn.Linear(hidden_size, config.max_action_dim)

    def forward(
        self,
        raw_features: list[Tensor],
        action_query_features: list[Tensor],
        raw_mask: Tensor,
        aq_mask: Tensor,
        proprio: Tensor,
        initial_actions: Tensor | None = None,
    ) -> Tensor:
        batch_size = proprio.shape[0]
        device = proprio.device
        if initial_actions is None:
            initial_actions = torch.zeros(
                batch_size,
                self.config.chunk_size,
                self.config.max_action_dim,
                device=device,
                dtype=proprio.dtype,
            )
        latent = self.initial_embed(initial_actions)
        proprio_token = self.proprio_proj(proprio)

        for idx, layer in enumerate(self.layers):
            raw = raw_features[min(idx, len(raw_features) - 1)].to(latent.dtype)
            aq = action_query_features[min(idx, len(action_query_features) - 1)].to(latent.dtype)
            latent = layer(latent, raw, aq, proprio_token, raw_mask, aq_mask)

        latent = self.final_norm(latent)
        actions = self.action_head(latent)
        return actions


# -----------------------------------------------------------------------------
# Full model
# -----------------------------------------------------------------------------


class VLAAdapterModel(nn.Module):
    def __init__(self, config: VLAAdapterConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = QwenBackbone(config)
        num_heads = config.policy_num_heads or self.backbone.num_attention_heads
        self.policy = BridgePolicy(
            config,
            hidden_size=self.backbone.hidden_size,
            num_layers=self.backbone.num_layers,
            num_heads=num_heads,
        )

    def forward(
        self,
        images: Iterable[Tensor],
        lang_tokens: Tensor,
        lang_attention_mask: Tensor,
        proprio: Tensor,
        initial_actions: Tensor | None = None,
    ) -> Tensor:
        backbone_out = self.backbone(images, lang_tokens, lang_attention_mask)
        actions = self.policy(
            backbone_out["raw_features"],
            backbone_out["action_query_features"],
            backbone_out["raw_attention_mask"],
            backbone_out["action_query_mask"],
            proprio,
            initial_actions,
        )
        return actions


# -----------------------------------------------------------------------------
# Policy wrapper
# -----------------------------------------------------------------------------


class VLAAdapterPolicy(PreTrainedPolicy):
    """Wrapper around :class:`VLAAdapterModel` to work within LeRobot."""

    config_class = VLAAdapterConfig
    name = "vla_adapter"

    def __init__(
        self,
        config: VLAAdapterConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ) -> None:
        super().__init__(config)
        config.validate_features()

        self.config = config
        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(config.output_features, config.normalization_mapping, dataset_stats)
        self.unnormalize_outputs = Unnormalize(config.output_features, config.normalization_mapping, dataset_stats)

        tokenizer_name = config.tokenizer_name or config.vlm_model_name
        self.language_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        if self.language_tokenizer.pad_token is None and self.language_tokenizer.eos_token is not None:
            self.language_tokenizer.pad_token = self.language_tokenizer.eos_token
        self.language_tokenizer.padding_side = "right"

        self.model = VLAAdapterModel(config)
        self.reset()

    # ------------------------------------------------------------------
    # Base API
    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._queues: dict[str, deque] = {ACTION: deque(maxlen=self.config.n_action_steps)}

    def get_optim_params(self) -> dict:
        return self.parameters()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        batch = self.normalize_inputs(batch)
        return batch

    def prepare_images(self, batch: dict[str, Tensor]) -> list[Tensor]:
        images = []
        keys = [k for k in self.config.image_features if k in batch]
        if not keys:
            raise ValueError("No image features found in the batch for VLA-Adapter policy.")

        target = self.config.resize_imgs_with_padding
        for key in keys:
            img = batch[key]
            if img.ndim == 5:
                img = img[:, -1]
            if target is not None:
                img = _resize_with_pad(img, target[0], target[1], pad_value=0)
            img = img * 2.0 - 1.0
            images.append(img)
        return images

    def prepare_language(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        device = next(self.model.parameters()).device
        tasks = batch.get("task", "")
        if isinstance(tasks, str):
            tasks = [tasks]
        if len(tasks) == 1:
            tasks = [tasks[0] for _ in range(batch[OBS_STATE].shape[0])]

        encoded = self.language_tokenizer(
            tasks,
            padding="longest",
            max_length=self.config.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )
        return encoded["input_ids"].to(device), encoded["attention_mask"].to(device)

    def prepare_state(self, batch: dict[str, Tensor]) -> Tensor:
        state = batch[OBS_STATE]
        state = state[:, -1, :] if state.ndim == 3 else state
        state = _pad_vector(state, self.config.max_state_dim)
        return state

    def prepare_action(self, batch: dict[str, Tensor]) -> Tensor:
        actions = batch[ACTION]
        if actions.ndim == 2:
            actions = actions[:, None, :]
        actions = _pad_vector(actions, self.config.max_action_dim)
        return actions[:, : self.config.chunk_size]

    # ------------------------------------------------------------------
    # Inference utilities
    # ------------------------------------------------------------------
    def _get_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        images = self.prepare_images(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)
        device = next(self.model.parameters()).device

        images = [img.to(device=device, dtype=torch.float32) for img in images]
        state = self.prepare_state(batch).to(device=device, dtype=torch.float32)

        initial = torch.zeros(
            state.shape[0],
            self.config.chunk_size,
            self.config.max_action_dim,
            device=device,
            dtype=torch.float32,
        )
        preds = self.model(images, lang_tokens, lang_masks, state, initial)
        preds = preds[:, :, : self.config.action_feature.shape[0]]
        preds = self.unnormalize_outputs({ACTION: preds})[ACTION]
        return preds

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:  # noqa: ARG002
        self.eval()
        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])
        return self._get_action_chunk(batch)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:  # noqa: ARG002
        self.eval()
        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])
        if len(self._queues[ACTION]) == 0:
            actions = self._get_action_chunk(batch)
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])
        return self._queues[ACTION].popleft()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def forward(self, batch: dict[str, Tensor], noise=None, time=None):  # noqa: D401, ARG002
        batch = self._prepare_batch(batch)
        batch = self.normalize_targets(batch)

        images = self.prepare_images(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)
        device = next(self.model.parameters()).device

        images = [img.to(device=device, dtype=torch.float32) for img in images]
        state = self.prepare_state(batch).to(device=device, dtype=torch.float32)
        actions = self.prepare_action(batch).to(device=device, dtype=torch.float32)

        initial = torch.zeros(
            state.shape[0],
            self.config.chunk_size,
            self.config.max_action_dim,
            device=device,
            dtype=torch.float32,
        )
        preds = self.model(images, lang_tokens, lang_masks, state, initial)
        preds = preds[:, : actions.shape[1]]
        loss = F.mse_loss(preds, actions, reduction="mean")
        return loss, {"loss": loss.item()}
