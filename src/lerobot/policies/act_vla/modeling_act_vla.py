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

from collections import deque
from typing import Optional

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from transformers import AutoModel, AutoProcessor

from lerobot.constants import ACTION, OBS_IMAGES, OBS_STATE
from lerobot.policies.act.modeling_act import (
    ACTDecoder,
    ACTEncoder,
    ACTTemporalEnsembler,
    create_sinusoidal_pos_embedding,
)
from lerobot.policies.act_vla.configuration_act_vla import ACTVLAConfig
from lerobot.policies.normalize import Normalize, Unnormalize
from lerobot.policies.pretrained import PreTrainedPolicy


def _resize_with_pad(img: Tensor, width: int, height: int, pad_value: float = -1.0) -> Tensor:
    """Resize while preserving aspect ratio, pad on left/top to target WxH.

    Args:
        img: (B, C, H, W)
    Returns:
        (B, C, H_pad, W_pad)
    """
    if img.ndim != 4:
        raise ValueError(f"(B,C,H,W) expected, got {img.shape}")

    cur_h, cur_w = img.shape[2:]
    ratio = max(cur_w / width, cur_h / height)
    new_h = int(cur_h / ratio)
    new_w = int(cur_w / ratio)
    resized = F.interpolate(img, size=(new_h, new_w), mode="bilinear", align_corners=False)
    pad_h = max(0, height - new_h)
    pad_w = max(0, width - new_w)
    # Pad (left, right, top, bottom) = (pad_w, 0, pad_h, 0)
    padded = F.pad(resized, (pad_w, 0, pad_h, 0), value=pad_value)
    return padded


class SiglipBackbone(nn.Module):
    """Minimal SigLIP backbone wrapper to embed images and language tokens."""

    def __init__(self, model_name: str, freeze_vision: bool = True):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        # Dims
        if hasattr(self.model.config, "vision_config"):
            self.vision_hidden_size = self.model.config.vision_config.hidden_size
        else:
            # Fallback
            self.vision_hidden_size = getattr(self.model, "hidden_size", None) or 1024

        if hasattr(self.model.config, "text_config"):
            self.text_hidden_size = self.model.config.text_config.hidden_size
        else:
            self.text_hidden_size = getattr(self.model, "hidden_size", None) or 768

        if freeze_vision and hasattr(self.model, "vision_model"):
            self.model.vision_model.eval()
            for p in self.model.vision_model.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def embed_image(self, pixel_values: Tensor) -> Tensor:
        # Expect pixel_values in [-1, 1]
        outputs = self.model.vision_model(pixel_values=pixel_values)
        hidden = outputs.last_hidden_state  # (B, N, C)
        return hidden

    def embed_language_tokens(self, input_ids: Tensor) -> Tensor:
        # Return token embeddings without running the text transformer
        return self.model.text_model.embeddings.token_embedding(input_ids)


class ReasoningEncoder(nn.Module):
    """Optional cross-modal reasoning encoder before ACT encoder."""

    def __init__(self, dim_model: int, n_layers: int, n_heads: int, dim_feedforward: int, dropout: float, activation: str):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=False,
        )
        self.encoder = nn.TransformerEncoder(self.layer, num_layers=n_layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class ACTVLA(nn.Module):
    """ACT backbone with SigLIP vision + language tokens and optional reasoning."""

    def __init__(self, config: ACTVLAConfig):
        super().__init__()
        self.config = config

        # No VAE in ACTVLA, latent will be zeros.

        # SigLIP backbone (vision+text)
        self.siglip = SiglipBackbone(
            model_name=self.config.vision_encoder_name,
            freeze_vision=self.config.freeze_vision_encoder,
        )

        # ACT encoder/decoder
        self.encoder = ACTEncoder(config)
        self.decoder = ACTDecoder(config)

        # Input projections to transformer hidden size
        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)
        if self.config.robot_state_feature:
            self.encoder_robot_state_input_proj = nn.Linear(
                self.config.robot_state_feature.shape[0], config.dim_model
            )
        if self.config.env_state_feature:
            self.encoder_env_state_input_proj = nn.Linear(
                self.config.env_state_feature.shape[0], config.dim_model
            )
        self.encoder_img_input_proj = nn.Linear(self.siglip.vision_hidden_size, config.dim_model)
        self.encoder_lang_input_proj = nn.Linear(self.siglip.text_hidden_size, config.dim_model)

        # Optional reasoning layer prior to ACT encoder
        self.reasoning = None
        if self.config.use_reasoning_vla:
            self.reasoning = ReasoningEncoder(
                dim_model=config.dim_model,
                n_layers=config.reasoning_layers,
                n_heads=config.reasoning_heads,
                dim_feedforward=config.reasoning_dim_feedforward,
                dropout=config.dropout,
                activation=config.feedforward_activation,
            )

        # Decoder positional embeddings and action head
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)
        self.action_head = nn.Linear(config.dim_model, self.config.action_feature.shape[0])

        self._reset_parameters()

    def _reset_parameters(self):
        # Xavier init for transformer parameters
        for p in list(self.encoder.parameters()) + list(self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _make_pos_embed(self, seq_len: int, batch_size: int, dim: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        pos = create_sinusoidal_pos_embedding(seq_len, dim).to(device=device, dtype=dtype)  # (S, D)
        return pos.unsqueeze(1).expand(-1, batch_size, -1)

    def _preprocess_images(self, images: list[Tensor]) -> list[Tensor]:
        preprocessed = []
        target = self.config.resize_imgs_with_padding
        for img in images:
            # img can be (B, C, H, W) or (B, T, C, H, W)
            if img.ndim == 5:
                if self.config.image_temporal_mode == "stack":
                    # Flatten time into batch for encoding, we'll unflatten implicitly by concatenation order
                    b, t, c, h, w = img.shape
                    frames = img.reshape(b * t, c, h, w)
                    if target is not None:
                        frames = _resize_with_pad(frames, target[0], target[1], pad_value=0)
                    frames = frames * 2.0 - 1.0
                    preprocessed.append(frames.reshape(b, t, c, target[1] if target else h, target[0] if target else w))
                else:  # last
                    img = img[:, -1]
                    if target is not None:
                        img = _resize_with_pad(img, target[0], target[1], pad_value=0)
                    img = img * 2.0 - 1.0
                    preprocessed.append(img)
            else:
                if target is not None:
                    img = _resize_with_pad(img, target[0], target[1], pad_value=0)
                img = img * 2.0 - 1.0
                preprocessed.append(img)
        return preprocessed

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Optional[Tensor], Optional[Tensor]]]:
        # batch may include: observation.state (B, D) or (B, T, D), observation.environment_state, observation.images (list of tensors),
        # lang_tokens
        if "observation.images" in batch:
            batch_size = batch["observation.images"][0].shape[0]
        elif OBS_STATE in batch:
            state = batch[OBS_STATE]
            batch_size = state.shape[0]
        else:
            # Fallback when only env_state provided
            batch_size = next(iter(batch.values())).shape[0]

        # Latent token: zeros (no VAE)
        device = batch[OBS_STATE].device if OBS_STATE in batch else next(iter(batch.values())).device
        latent = torch.zeros(batch_size, self.config.latent_dim, dtype=torch.float32, device=device)

        encoder_tokens: list[Tensor] = []  # (S, B, C)
        encoder_pos: list[Tensor] = []     # (S, B, C)

        # Latent token
        latent_tok = self.encoder_latent_input_proj(latent).unsqueeze(0)  # (1, B, C)
        latent_pos = torch.zeros_like(latent_tok)
        encoder_tokens.append(latent_tok)
        encoder_pos.append(latent_pos)

        # Robot state tokens
        if self.config.robot_state_feature and OBS_STATE in batch:
            state = batch[OBS_STATE]
            if state.ndim == 3 and self.config.state_temporal_mode == "stack":
                b, t, d = state.shape
                state_proj = self.encoder_robot_state_input_proj(state.reshape(b * t, d)).reshape(b, t, -1)
                state_proj = state_proj.transpose(0, 1)  # (T, B, C)
                pos = self._make_pos_embed(state_proj.shape[0], b, state_proj.shape[-1], state_proj.device, state_proj.dtype)
                encoder_tokens.append(state_proj)
                encoder_pos.append(pos)
            else:
                if state.ndim == 3:
                    state = state.mean(dim=1)
                state_proj = self.encoder_robot_state_input_proj(state).unsqueeze(0)
                pos = torch.zeros_like(state_proj)
                encoder_tokens.append(state_proj)
                encoder_pos.append(pos)

        # Env state token
        if self.config.env_state_feature and "observation.environment_state" in batch:
            env_state = batch["observation.environment_state"]
            if env_state.ndim == 3:
                env_state = env_state.mean(dim=1)
            env_proj = self.encoder_env_state_input_proj(env_state).unsqueeze(0)
            pos = torch.zeros_like(env_proj)
            encoder_tokens.append(env_proj)
            encoder_pos.append(pos)

        # Language tokens
        if self.config.use_language and ("lang_tokens" in batch):
            lang_emb = self.siglip.embed_language_tokens(batch["lang_tokens"])  # (B, L, D_t)
            lang_proj = self.encoder_lang_input_proj(lang_emb)  # (B, L, C)
            lang_proj = lang_proj.transpose(0, 1)  # (L, B, C)
            pos = self._make_pos_embed(lang_proj.shape[0], batch_size, lang_proj.shape[-1], lang_proj.device, lang_proj.dtype)
            encoder_tokens.append(lang_proj)
            encoder_pos.append(pos)

        # Image tokens (per camera)
        if "observation.images" in batch:
            images_list = self._preprocess_images(batch["observation.images"])  # list of (B,C,H,W) or (B,T,C,H,W)
            for img in images_list:
                if img.ndim == 5:  # (B, T, C, H, W)
                    b, t, c, h, w = img.shape
                    img_flat = img.reshape(b * t, c, h, w)
                    img_emb = self.siglip.embed_image(img_flat)  # (B*T, N, D_v)
                    n = img_emb.shape[1]
                    img_emb = img_emb.reshape(b, t * n, -1)
                else:
                    img_emb = self.siglip.embed_image(img)  # (B, N, D_v)
                img_proj = self.encoder_img_input_proj(img_emb)  # (B, N', C)
                img_proj = img_proj.transpose(0, 1)  # (N', B, C)
                pos = self._make_pos_embed(img_proj.shape[0], batch_size, img_proj.shape[-1], img_proj.device, img_proj.dtype)
                encoder_tokens.append(img_proj)
                encoder_pos.append(pos)

        # Concatenate all tokens/pos
        enc_in = torch.cat(encoder_tokens, dim=0)
        enc_pos = torch.cat(encoder_pos, dim=0)

        # Optional reasoning pass
        if self.reasoning is not None:
            enc_in = self.reasoning(enc_in)

        # ACT encoder/decoder
        enc_out = self.encoder(enc_in, pos_embed=enc_pos)
        dec_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model), dtype=enc_out.dtype, device=enc_out.device
        )
        dec_out = self.decoder(
            dec_in,
            enc_out,
            encoder_pos_embed=enc_pos,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )
        dec_out = dec_out.transpose(0, 1)
        actions = self.action_head(dec_out)
        return actions, (None, None)


class ACTVLAPolicy(PreTrainedPolicy):
    """ACT policy wrapper with SigLIP vision + language and temporal history."""

    config_class = ACTVLAConfig
    name = "act_vla"

    def __init__(self, config: ACTVLAConfig, dataset_stats: dict[str, dict[str, Tensor]] | None = None):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(config.output_features, config.normalization_mapping, dataset_stats)
        self.unnormalize_outputs = Unnormalize(config.output_features, config.normalization_mapping, dataset_stats)

        # Tokenizer for language (only tokenizer, embeddings come from SigLIP text model)
        if self.config.use_language:
            self.language_tokenizer = AutoProcessor.from_pretrained(self.config.language_tokenizer_name).tokenizer
        else:
            self.language_tokenizer = None

        self.model = ACTVLA(config)

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

        self.reset()

    def get_optim_params(self) -> dict:
        return self.parameters()

    def reset(self):
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)
        # History queues for observations
        self._queues = {
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
        }

    def _prepare_language(self, batch: dict[str, Tensor]) -> Optional[Tensor]:
        if not self.config.use_language or self.language_tokenizer is None:
            return None
        device = next(iter(batch.values())).device
        task = batch.get("task") or batch.get("instruction")
        if task is None:
            return None
        tasks = task if isinstance(task, list) else [task]
        if len(tasks) == 1:
            bsize = batch[OBS_STATE].shape[0] if OBS_STATE in batch else next(iter(batch.values())).shape[0]
            tasks = [tasks[0] for _ in range(bsize)]
        tasks = [t if t.endswith("\n") else f"{t}\n" for t in tasks]
        tokenized = self.language_tokenizer.__call__(
            tasks,
            padding=self.config.pad_language_to,
            padding_side="right",
            max_length=self.config.tokenizer_max_length,
            return_tensors="pt",
        )
        lang_tokens = tokenized["input_ids"].to(device=device)
        return lang_tokens

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        if self.config.temporal_ensemble_coeff is not None:
            actions = self.predict_action_chunk(batch)
            return self.temporal_ensembler.update(actions)
        if len(getattr(self, "_action_queue", [])) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        batch = self.normalize_inputs(batch)
        # Populate state history queue and expose as time-major tensors
        from lerobot.policies.utils import populate_queues
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])
        # Stack to shape (B, n_obs_steps, D) if we have state
        if OBS_STATE in self._queues and len(self._queues[OBS_STATE]) > 0:
            # queues store individual tensors shaped (B, D); stack them
            state_hist = torch.stack(list(self._queues[OBS_STATE]), dim=1)
            batch = dict(batch)
            batch[OBS_STATE] = state_hist
        if self.config.image_features:
            b = dict(batch)
            b[OBS_IMAGES] = [b[key] for key in self.config.image_features]
        else:
            b = batch
        lang_tokens = self._prepare_language(b)
        if lang_tokens is not None:
            b = dict(b)
            b["lang_tokens"] = lang_tokens
        actions = self.model(b)[0]
        actions = self.unnormalize_outputs({ACTION: actions})[ACTION]
        return actions

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        lang_tokens = self._prepare_language(batch)
        if lang_tokens is not None:
            batch = dict(batch)
            batch["lang_tokens"] = lang_tokens
        batch = self.normalize_targets(batch)
        actions_hat, (_mu_hat, _log_sigma_x2_hat) = self.model(batch)

        l1_loss = (
            F.l1_loss(batch[ACTION], actions_hat, reduction="none") * ~batch.get("action_is_pad", torch.zeros_like(batch[ACTION][..., 0], dtype=torch.bool)).unsqueeze(-1)
        ).mean()

        loss_dict = {"l1_loss": l1_loss.item()}
        loss = l1_loss

        return loss, loss_dict


