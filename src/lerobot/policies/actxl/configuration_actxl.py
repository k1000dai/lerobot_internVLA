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

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig


@PreTrainedConfig.register_subclass("actxl")
@dataclass
class ACTXLConfig(PreTrainedConfig):
    """Configuration for ACT-XL: ACT with SigLIP vision and language tokens (no VAE/history).

    Key additions over ACT:
      - vision_encoder_type: choose between torchvision ResNet or transformers SigLIP.
      - language_model_name_or_path: optional text encoder to inject task language tokens.
      - large default transformer dimensions to scale towards 1B-class capacity.
    """

    # IO structure
    n_obs_steps: int = 1  # kept for compatibility; state history handled via state_history_steps
    chunk_size: int = 50
    n_action_steps: int = 50

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
            "ENV": NormalizationMode.MEAN_STD,
        }
    )

    # Vision encoder
    vision_encoder_type: str = "siglip"  # "siglip" | "resnet"
    # When using siglip, set this to a local or hub path
    vision_model_name_or_path: str = "google/siglip-base-patch16-224"
    freeze_vision_encoder: bool = True

    # Legacy torchvision ResNet path (if chosen)
    vision_backbone: str = "resnet50"
    pretrained_backbone_weights: str | None = "ResNet50_Weights.IMAGENET1K_V2"
    replace_final_stride_with_dilation: bool = False

    # Language encoder (optional)
    language_model_name_or_path: str | None = "google/siglip-base-patch16-224"
    tokenizer_max_length: int = 128
    pad_language_to: str | bool = "longest"  # pad strategy for tokenizer
    freeze_text_encoder: bool = True

    # Transformer dimensions (scaled up towards ~1B-class)
    pre_norm: bool = True
    dim_model: int = 1024
    n_heads: int = 16
    dim_feedforward: int = 8192
    # Activation used in transformer feed-forward layers (matches ACT expectations)
    feedforward_activation: str = "gelu"
    n_encoder_layers: int = 24
    # We allow multiple decoder layers; implementation uses them all
    n_decoder_layers: int = 8

    # Regularization / loss
    dropout: float = 0.1

    # Reasoning tokens (lightweight planning)
    reasoning_num_tokens: int = 8  # default on; set to 0 to disable
    reasoning_layers: int = 2
    reasoning_source: str = "language"  # "language" | "lang+vision"
    reasoning_dim_feedforward: int = 4096

    # Inference
    temporal_ensemble_coeff: float | None = None

    # Training preset
    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 1e-5

    def __post_init__(self):
        super().__post_init__()

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) > chunk_size ({self.chunk_size})."
            )
        if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
            raise NotImplementedError(
                "When using temporal ensembling, set n_action_steps=1 to query the policy each step."
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        # Accepts vision and/or env state, robot state optional (but recommended)
        if not self.image_features and not self.env_state_feature and not self.robot_state_feature:
            raise ValueError(
                "Provide at least one of: image features, env state, or robot state among the inputs."
            )

    @property
    def observation_delta_indices(self) -> list | None:
        return None

    @property
    def action_delta_indices(self) -> list | None:
        # Return list of chunk indices to mark action deltas (full horizon)
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> list | None:
        return None
