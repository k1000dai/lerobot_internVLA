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

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig


@PreTrainedConfig.register_subclass("act_vla")
@dataclass
class ACTVLAConfig(PreTrainedConfig):
    """Configuration class for ACT-VLA: ACT with SigLIP vision and language tokens.

    Adds:
      - HF SigLIP vision encoder via `vision_encoder_name`
      - Language token pathway via tokenizer `language_tokenizer_name`
      - Temporal history for state/images via `n_obs_steps`
      - Optional reasoning cross-modal transformer (`use_reasoning_vla`)
      - Depth/width scaling presets for large models (~1B params)
    """

    # Input / output structure
    n_obs_steps: int = 2
    chunk_size: int = 50
    n_action_steps: int = 50

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Vision encoder (HF transformers)
    vision_encoder_name: str | None = "google/siglip-base-patch16-224"
    freeze_vision_encoder: bool = True
    # Optional resize to encoder's native res with padding (W, H). None to disable.
    resize_imgs_with_padding: tuple[int, int] | None = (224, 224)

    # Language pathway
    use_language: bool = True
    language_tokenizer_name: str = "google/siglip-base-patch16-224"
    tokenizer_max_length: int = 64
    pad_language_to: str = "longest"  # or "max_length"

    # How to incorporate temporal history from observations (T = n_obs_steps)
    # images: "last" uses most recent frame; "stack" appends tokens from T frames
    image_temporal_mode: str = "last"
    # state: "stack" adds T 1D tokens; "mean" averages over time then 1 token
    state_temporal_mode: str = "stack"

    # Transformer architecture
    pre_norm: bool = False
    dim_model: int = 768
    n_heads: int = 12
    dim_feedforward: int = 3072
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 8
    n_decoder_layers: int = 2
    dropout: float = 0.1

    # Latent token dimension for the decoder queries (not a VAE latent)
    latent_dim: int = 64

    # Optional temporal ensemble at inference (same constraint as ACT)
    temporal_ensemble_coeff: float | None = None

    # Reasoning VLA cross-modal encoder prior to ACT encoder
    use_reasoning_vla: bool = False
    reasoning_layers: int = 4
    reasoning_heads: int = 8
    reasoning_dim_feedforward: int = 4096

    # Training preset (align with SmolVLA defaults)
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10
    optimizer_lr_backbone: float = 1e-5

    # Scheduler (SmolVLA-style cosine with warmup)
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    # Convenient preset to scale up model size; if set, overrides dims above.
    # Choices: None | "base" | "large" | "xl" | "approx_1b"
    model_scale: str | None = None

    def __post_init__(self):
        super().__post_init__()

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
            raise NotImplementedError(
                "`n_action_steps` must be 1 when using temporal ensembling. This is "
                "because the policy needs to be queried every step to compute the ensembled action."
            )
        if self.image_temporal_mode not in ("last", "stack"):
            raise ValueError("image_temporal_mode must be 'last' or 'stack'")
        if self.state_temporal_mode not in ("stack", "mean"):
            raise ValueError("state_temporal_mode must be 'stack' or 'mean'")

        # Apply scale presets if requested
        if self.model_scale is not None:
            preset = self.model_scale
            if preset == "base":
                self.dim_model = 768
                self.n_heads = 12
                self.dim_feedforward = 3072
                self.n_encoder_layers = 12
                self.n_decoder_layers = 4
            elif preset == "large":
                self.dim_model = 1024
                self.n_heads = 16
                self.dim_feedforward = 4096
                self.n_encoder_layers = 24
                self.n_decoder_layers = 6
            elif preset == "xl":
                self.dim_model = 1536
                self.n_heads = 24
                self.dim_feedforward = 6144
                self.n_encoder_layers = 36
                self.n_decoder_layers = 8
            elif preset == "approx_1b":
                # Roughly ~1B class depending on inputs; exact count varies
                self.dim_model = 2048
                self.n_heads = 32
                self.dim_feedforward = 8192
                self.n_encoder_layers = 48
                self.n_decoder_layers = 12
            else:
                raise ValueError(f"Unknown model_scale preset: {preset}")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature and not self.robot_state_feature:
            raise ValueError(
                "You must provide at least one of: image, robot state, or environment state among the inputs."
            )

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None


