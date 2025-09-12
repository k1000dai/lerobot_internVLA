#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""ACT-XL Policy: ACT with SigLIP vision and language conditioning.

This implementation mirrors the original ACT structure while:
- swapping the vision encoder for SigLIP (via transformers) when requested,
- adding a text encoder and language token embeddings to the encoder sequence,
- scaling transformer depth/width for ~1B-class capacity,
- simplifying by removing VAE and state-history handling.
"""

from collections import deque
from itertools import chain
from typing import Iterable

import einops
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lerobot.constants import ACTION, OBS_IMAGES, OBS_STATE
from lerobot.policies.act.modeling_act import ACTDecoder, ACTEncoder, ACTTemporalEnsembler, create_sinusoidal_pos_embedding
from lerobot.policies.actxl.configuration_actxl import ACTXLConfig
from lerobot.policies.normalize import Normalize, Unnormalize
from lerobot.policies.pretrained import PreTrainedPolicy


# Optional transformers imports (SigLIP + text encoder). Keep imports lazy-safe.
try:
    from transformers import (
        AutoTokenizer,
        AutoModel,
        SiglipVisionModel,
        SiglipTextModel,
        SiglipImageProcessor,
    )
except Exception:  # pragma: no cover - transformers optional at import time
    AutoTokenizer = None
    AutoModel = None
    SiglipVisionModel = None
    SiglipTextModel = None
    SiglipImageProcessor = None


class ACTXLPolicy(PreTrainedPolicy):
    config_class = ACTXLConfig
    name = "actxl"

    def __init__(self, config: ACTXLConfig, dataset_stats: dict[str, dict[str, Tensor]] | None = None):
        super().__init__(config)
        config.validate_features()
        self.config = config

        # For SigLIP, skip dataset-based normalization on images to avoid double-normalizing.
        from lerobot.configs.types import NormalizationMode, FeatureType

        input_norm_map = dict(config.normalization_mapping)
        if getattr(config, "vision_encoder_type", "siglip") == "siglip":
            input_norm_map[FeatureType.VISUAL.value] = NormalizationMode.IDENTITY

        self.normalize_inputs = Normalize(config.input_features, input_norm_map, dataset_stats)
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )

        # Tokenizer for language, if enabled
        self.language_tokenizer = None
        if self.config.language_model_name_or_path and AutoTokenizer is not None:
            try:
                self.language_tokenizer = AutoTokenizer.from_pretrained(
                    self.config.language_model_name_or_path
                )
            except Exception:
                self.language_tokenizer = None

        self.model = ACTXL(config)

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

        self.reset()

    def get_optim_params(self) -> dict:
        return [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith("model.vision_encoder") and not n.startswith("model.text_encoder") and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if (n.startswith("model.vision_encoder") or n.startswith("model.text_encoder")) and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    def reset(self):
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue: deque[Tensor] = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()

        # No history management: if a temporal dimension is provided, we will take the last step in forward

        if self.config.temporal_ensemble_coeff is not None:
            actions = self.predict_action_chunk(batch)
            action = self.temporal_ensembler.update(actions)
            return action

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        batch = self.normalize_inputs(batch)

        # Collect vision inputs
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        # Prepare language tokens if tokenizer available and text present
        if self.language_tokenizer is not None and ("task" in batch or "language_instruction" in batch):
            tokens, masks = self.prepare_language(batch)
            batch = dict(batch)
            batch["language.input_ids"] = tokens
            batch["language.attention_mask"] = masks

        actions = self.model(batch)[0]
        actions = self.unnormalize_outputs({ACTION: actions})[ACTION]
        return actions

    def prepare_language(self, batch) -> tuple[Tensor, Tensor]:
        device = next(self.parameters()).device
        tasks = batch.get("task", batch.get("language_instruction", ""))
        if isinstance(tasks, str):
            tasks = [tasks]
        if len(tasks) == 1:
            # Broadcast to batch if single string
            # Try infer batch size from state/images
            if OBS_STATE in batch:
                bsize = batch[OBS_STATE].shape[0] if batch[OBS_STATE].ndim == 2 else batch[OBS_STATE].shape[0]
            elif OBS_IMAGES in batch and len(batch[OBS_IMAGES]) > 0:
                bsize = batch[OBS_IMAGES][0].shape[0]
            else:
                bsize = 1
            tasks = [tasks[0] for _ in range(bsize)]

        tokenized = self.language_tokenizer.__call__(
            tasks,
            padding=self.config.pad_language_to,
            padding_side="right",
            max_length=self.config.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"].to(device)
        attn_mask = tokenized["attention_mask"].to(device, dtype=torch.bool)
        return input_ids, attn_mask

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        batch = self.normalize_inputs(batch)

        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        batch = self.normalize_targets(batch)
        actions_hat, _ = self.model(batch)

        l1 = (F.l1_loss(batch[ACTION], actions_hat, reduction="none") * ~batch["action_is_pad"].unsqueeze(-1)).mean()

        loss_dict = {"l1_loss": l1.item()}
        loss = l1
        return loss, loss_dict



class ACTXL(nn.Module):
    """ACT backbone with SigLIP vision + text. Simpler variant without VAE or state history."""

    def __init__(self, config: ACTXLConfig):
        super().__init__()
        self.config = config

        # Learned latent token (replaces ACT VAE latent)
        self.latent_token = nn.Parameter(torch.zeros(1, config.dim_model))
        nn.init.normal_(self.latent_token, mean=0.0, std=0.02)

        # Vision encoder
        self.vision_encoder = None
        self.vision_processor = None
        self.vision_hidden_size = None
        if self.config.image_features:
            if self.config.vision_encoder_type == "siglip" and SiglipVisionModel is not None:
                self.vision_encoder = SiglipVisionModel.from_pretrained(self.config.vision_model_name_or_path)
                try:
                    self.vision_processor = SiglipImageProcessor.from_pretrained(
                        self.config.vision_model_name_or_path
                    )
                except Exception:
                    self.vision_processor = None
                self.vision_hidden_size = int(self.vision_encoder.config.hidden_size)
                if self.config.freeze_vision_encoder:
                    for p in self.vision_encoder.parameters():
                        p.requires_grad = False
            else:
                # Fallback to ACT-style ResNet encoder: use last feature map and 1x1 conv projection
                import torchvision
                from torchvision.models._utils import IntermediateLayerGetter
                from torchvision.ops.misc import FrozenBatchNorm2d

                backbone_model = getattr(torchvision.models, self.config.vision_backbone)(
                    replace_stride_with_dilation=[
                        False,
                        False,
                        self.config.replace_final_stride_with_dilation,
                    ],
                    weights=self.config.pretrained_backbone_weights,
                    norm_layer=FrozenBatchNorm2d,
                )
                self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})
                self.encoder_img_feat_input_proj = nn.Conv2d(backbone_model.fc.in_features, config.dim_model, 1)

        # Text encoder (optional)
        self.text_encoder = None
        self.text_hidden_size = None
        if self.config.language_model_name_or_path and (SiglipTextModel is not None or AutoModel is not None):
            try:
                if SiglipTextModel is not None:
                    self.text_encoder = SiglipTextModel.from_pretrained(self.config.language_model_name_or_path)
                    self.text_hidden_size = int(self.text_encoder.config.hidden_size)
                else:
                    self.text_encoder = AutoModel.from_pretrained(self.config.language_model_name_or_path)
                    self.text_hidden_size = int(self.text_encoder.config.hidden_size)
                if self.config.freeze_text_encoder:
                    for p in self.text_encoder.parameters():
                        p.requires_grad = False
            except Exception:
                self.text_encoder = None

        # Transformer (encoder/decoder)
        self.encoder = ACTEncoder(config)
        self.decoder = ACTDecoder(config)

        # Reasoning module (optional): produces a small set of plan tokens
        if self.config.reasoning_num_tokens > 0:
            self.reasoner = Reasoner(
                dim_model=config.dim_model,
                n_heads=config.n_heads,
                dim_feedforward=getattr(config, "reasoning_dim_feedforward", config.dim_feedforward),
                n_layers=config.reasoning_layers,
                n_queries=config.reasoning_num_tokens,
                dropout=config.dropout,
                pre_norm=config.pre_norm,
            )
            self.reasoning_pos = nn.Embedding(config.reasoning_num_tokens, config.dim_model)
        else:
            self.reasoner = None

        # Input projections to transformer model dimension
        if self.config.robot_state_feature:
            self.encoder_state_input_proj = nn.Linear(self.config.robot_state_feature.shape[0], config.dim_model)
        if self.config.env_state_feature:
            self.encoder_env_state_input_proj = nn.Linear(self.config.env_state_feature.shape[0], config.dim_model)
        if self.config.image_features and self.config.vision_encoder_type == "siglip":
            assert self.vision_hidden_size is not None
            self.encoder_img_tok_input_proj = nn.Linear(self.vision_hidden_size, config.dim_model)
        if self.text_encoder is not None and self.text_hidden_size is not None:
            self.encoder_lang_tok_input_proj = nn.Linear(self.text_hidden_size, config.dim_model)

        # Positional embeddings for various token groups
        self.latent_pos = nn.Embedding(1, config.dim_model)
        # Single state token position
        self.state_pos = nn.Embedding(1, config.dim_model)
        # Language positions up to tokenizer_max_length
        self.lang_pos = nn.Embedding(max(self.config.tokenizer_max_length, 1), config.dim_model)
        # No reasoning queries: keep encoder input minimal

        # Decoder positional embeddings and action head
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)
        self.action_head = nn.Linear(config.dim_model, self.config.action_feature.shape[0])

        self._reset_parameters()

    def _reset_parameters(self):
        modules = [self.encoder, self.decoder]
        if getattr(self, "reasoner", None) is not None:
            modules.append(self.reasoner)
        for p in chain.from_iterable(m.parameters() for m in modules):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _preprocess_for_siglip(self, img: Tensor) -> Tensor:
        """Resize and normalize an image batch (B,3,H,W) for SigLIP vision encoder.

        Prefers the transformers SiglipImageProcessor if available; otherwise falls back to
        CLIP-style mean/std normalization after bilinear resize.
        """
        device = next(self.parameters()).device
        if self.vision_encoder is not None:
            try:
                dtype = next(self.vision_encoder.parameters()).dtype
            except StopIteration:
                dtype = img.dtype
        else:
            dtype = img.dtype
        target = getattr(getattr(self.vision_encoder, "config", None), "image_size", 384)

        if self.vision_processor is not None:
            # Convert (B,C,H,W) [0,1] -> list of HWC uint8 for processor
            img_cpu = img.detach().cpu()
            imgs_hwc = einops.rearrange(img_cpu.clamp(0, 1), "b c h w -> b h w c")
            imgs_uint8 = (imgs_hwc * 255.0).to(torch.uint8)
            outputs = self.vision_processor(images=[x.numpy() for x in imgs_uint8], return_tensors="pt")
            pixel_values = outputs["pixel_values"].to(device=device, dtype=dtype)
            return pixel_values

        # Manual fallback: bilinear resize + CLIP mean/std
        x = torch.nn.functional.interpolate(img, size=(target, target), mode="bilinear", align_corners=False)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        x = (x - mean) / std
        return x.to(device=device, dtype=dtype)

    def _encode_images(self, images: Iterable[Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Return per-camera token sequences and (optional) pos embeddings as lists of (S, B, D)."""
        img_tok_lists: list[Tensor] = []
        pos_tok_lists: list[Tensor] = []

        if self.config.vision_encoder_type == "siglip" and self.vision_encoder is not None:
            for img in images:
                pixel_values = self._preprocess_for_siglip(img)
                vision_out = self.vision_encoder(pixel_values=pixel_values)
                # Use patch tokens (drop cls) -> (B, N, H)
                seq = vision_out.last_hidden_state[:, 1:, :]
                seq = self.encoder_img_tok_input_proj(seq)
                seq = einops.rearrange(seq, "b s d -> s b d")
                img_tok_lists.append(seq)
                pos_tok_lists.append(None)
        else:
            # Fallback to ACT-style conv feature map flattened to tokens
            assert hasattr(self, "backbone") and hasattr(self, "encoder_img_feat_input_proj")
            for img in images:
                cam_features = self.backbone(img)["feature_map"]  # (B, C, H, W)
                cam_features = self.encoder_img_feat_input_proj(cam_features)
                cam_pos_embed = create_sinusoidal_pos_embedding(cam_features.shape[-2] * cam_features.shape[-1], self.config.dim_model)
                cam_pos_embed = cam_pos_embed.to(device=cam_features.device, dtype=cam_features.dtype).unsqueeze(1)
                seq = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                img_tok_lists.append(seq)
                pos_tok_lists.append(cam_pos_embed)
        return img_tok_lists, pos_tok_lists

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        # Determine batch size from available keys
        if "observation.images" in batch:
            batch_size = batch["observation.images"][0].shape[0]
        else:
            batch_size = batch[OBS_STATE].shape[0]

        # No VAE: use a learned latent token repeated for the batch
        mu = log_sigma_x2 = None
        latent_tok = self.latent_token.expand(batch_size, -1)  # (B, D)

        # Assemble encoder tokens and positional embeddings (as lists)
        enc_tokens: list[Tensor] = []
        enc_pos: list[Tensor | None] = []

        # Latent token
        enc_tokens.append(latent_tok)  # (B, D)
        enc_pos.append(self.latent_pos.weight[0:1].unsqueeze(1))  # (1,1,D)

        # Robot state token: single step (take last if temporal)
        if OBS_STATE in batch:
            state_in = batch[OBS_STATE]
            if state_in.ndim > 2:
                state_in = state_in[:, -1]
            state_tok = self.encoder_state_input_proj(state_in) if self.config.robot_state_feature else state_in
            enc_tokens.append(state_tok)
            enc_pos.append(self.state_pos.weight[:1].unsqueeze(1))

        # Env state token (optional)
        if self.config.env_state_feature and "observation.environment_state" in batch:
            env_tok = self.encoder_env_state_input_proj(batch["observation.environment_state"])  # (B, D)
            enc_tokens.append(env_tok)
            enc_pos.append(None)

        # Prepare language tokens (optional) but defer adding to sequence
        lang_seq: Tensor | None = None  # (L,B,D)
        lang_pos: Tensor | None = None  # (L,1,D)
        if self.text_encoder is not None and "language.input_ids" in batch:
            ids = batch["language.input_ids"]
            attn = batch.get("language.attention_mask", None)
            if hasattr(self.text_encoder, "forward"):
                text_out = self.text_encoder(input_ids=ids, attention_mask=attn)
                hid = text_out.last_hidden_state  # (B, L, H)
                hid = self.encoder_lang_tok_input_proj(hid)  # (B, L, D)
                L = hid.shape[1]
                lang_pos = self.lang_pos.weight[:L].unsqueeze(1)  # (L,1,D)
                lang_seq = einops.rearrange(hid, "b l d -> l b d")  # (L,B,D)

        # Prepare vision tokens per camera (defer adding)
        img_tok_lists: list[Tensor] | None = None
        pos_tok_lists: list[Tensor] | None = None
        if self.config.image_features and OBS_IMAGES in batch:
            img_tok_lists, pos_tok_lists = self._encode_images(batch[OBS_IMAGES])

        # Reasoning tokens (optional): derive plan tokens from language/vision context
        if self.reasoner is not None:
            context_parts: list[Tensor] = []
            src = getattr(self.config, "reasoning_source", "language")
            if (src == "language" or src == "lang+vision") and lang_seq is not None:
                context_parts.append(lang_seq)
            if src == "lang+vision" and img_tok_lists is not None and len(img_tok_lists) > 0:
                context_parts.extend(img_tok_lists)

            if len(context_parts) == 0:
                # Fallback context: state token or latent token
                if len(enc_tokens) >= 2:
                    # enc_tokens[1] should be state when present
                    context_parts = [enc_tokens[1].unsqueeze(0)]
                else:
                    context_parts = [enc_tokens[0].unsqueeze(0)]

            context = torch.cat(context_parts, dim=0)  # (S,B,D)
            reasoning_seq = self.reasoner(context)  # (R,B,D)
            R = reasoning_seq.shape[0]
            enc_tokens.extend(list(reasoning_seq))
            # Learned positional embeddings for reasoning tokens
            enc_pos.extend(list(self.reasoning_pos.weight[:R].unsqueeze(1)))

        # Now append language and vision tokens to the encoder input
        if lang_seq is not None and lang_pos is not None:
            enc_tokens.extend(list(lang_seq))
            enc_pos.extend(list(lang_pos))

        if img_tok_lists is not None and pos_tok_lists is not None:
            for seq, pos in zip(img_tok_lists, pos_tok_lists):
                enc_tokens.extend(list(seq))
                if pos is None:
                    enc_pos.extend([None] * seq.shape[0])
                else:
                    pos = pos.to(device=seq.device, dtype=seq.dtype)
                    enc_pos.extend(list(pos))

        # Stack all tokens/positions -> (S,B,D)
        enc_tokens_t = torch.stack(enc_tokens, dim=0)
        # Build combined pos embedding tensor; replace None with zeros
        pos_elems = [p if p is not None else torch.zeros((1, 1, self.config.dim_model), device=enc_tokens_t.device, dtype=enc_tokens_t.dtype) for p in enc_pos]
        enc_pos_t = torch.cat(pos_elems, dim=0)

        # Encode/Decode
        encoder_out = self.encoder(enc_tokens_t, pos_embed=enc_pos_t)
        decoder_in = torch.zeros(
            (self.config.chunk_size, encoder_out.shape[1], self.config.dim_model),
            dtype=encoder_out.dtype,
            device=encoder_out.device,
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=enc_pos_t,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )
        decoder_out = decoder_out.transpose(0, 1)  # (B,S,D)
        actions = self.action_head(decoder_out)
        return actions, (mu, log_sigma_x2)


class ReasoningBlock(nn.Module):
    """A lightweight cross-attention block operating on a small set of queries.

    The queries attend to a provided context (e.g., language and/or vision tokens)
    and pass through a feed-forward MLP. Pre-norm or post-norm matches ACT config.
    """

    def __init__(
        self,
        dim_model: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float,
        pre_norm: bool,
    ) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim_model, num_heads=n_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_model),
        )
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.pre_norm = pre_norm

    def forward(self, queries: Tensor, context: Tensor) -> Tensor:
        # queries: (R,B,D), context: (S,B,D)
        x = queries
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        # Cross attention: queries attend to context
        x = self.cross_attn(query=x, key=context, value=context)[0]
        x = skip + self.dropout1(x)

        skip = x
        if self.pre_norm:
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.ffn(x)
        x = skip + self.dropout2(x)
        if not self.pre_norm:
            x = self.norm2(x)
        return x


class Reasoner(nn.Module):
    """Produces a set of reasoning/plan tokens from context via stacked blocks."""

    def __init__(
        self,
        *,
        dim_model: int,
        n_heads: int,
        dim_feedforward: int,
        n_layers: int,
        n_queries: int,
        dropout: float,
        pre_norm: bool,
    ) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.zeros(n_queries, 1, dim_model))  # (R,1,D)
        nn.init.normal_(self.queries, mean=0.0, std=0.02)
        self.layers = nn.ModuleList(
            [
                ReasoningBlock(
                    dim_model=dim_model,
                    n_heads=n_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    pre_norm=pre_norm,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, context: Tensor) -> Tensor:
        # context: (S,B,D)
        R, _, D = self.queries.shape
        B = context.shape[1]
        q = self.queries.expand(R, B, D)
        for layer in self.layers:
            q = layer(q, context)
        return q  # (R,B,D)
