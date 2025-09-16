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
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig


@PreTrainedConfig.register_subclass("vla_adapter")
@dataclass
class VLAAdapterConfig(PreTrainedConfig):
    """Configuration for the VLA-Adapter policy."""

    # Input / output structure
    n_obs_steps: int = 1
    chunk_size: int = 8
    n_action_steps: int = 8

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    max_state_dim: int = 64
    max_action_dim: int = 32

    # Image preprocessing
    resize_imgs_with_padding: tuple[int, int] | None = (224, 224)

    # Tokenization / backbone
    tokenizer_name: str | None = None
    tokenizer_max_length: int = 128
    vlm_model_name: str = "Qwen/Qwen2.5-0.5B"
    freeze_vlm: bool = False
    patch_size: int = 14
    num_action_queries: int = 64

    # Policy (Bridge Attention)
    policy_hidden_size: int = 384
    policy_num_layers: int | None = None
    policy_num_heads: int = 8
    policy_mlp_ratio: float = 4.0
    bridge_ratio_init: float = 0.0

    # Training hyper-parameters
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-4
    optimizer_grad_clip_norm: float = 1.0

    scheduler_warmup_steps: int = 2_000
    scheduler_decay_steps: int = 50_000
    scheduler_decay_lr: float = 1e-5

    use_delta_joint_actions: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                "The number of action steps cannot exceed the chunk size: "
                f"n_action_steps={self.n_action_steps}, chunk_size={self.chunk_size}."
            )
        if self.num_action_queries <= 0:
            raise ValueError("num_action_queries must be positive for VLA-Adapter.")
        if self.policy_hidden_size <= 0:
            raise ValueError("policy_hidden_size must be positive.")
        if self.policy_num_layers is not None and self.policy_num_layers <= 0:
            raise ValueError("policy_num_layers must be positive when provided.")

    def validate_features(self) -> None:
        # Ensure proprioceptive / state feature exists when declared in config
        if self.max_state_dim > 0 and not any(ft.type is FeatureType.STATE for ft in self.input_features.values()):
            # Create a placeholder state feature if absent so that normalization modules are well formed.
            self.input_features.setdefault(
                "observation.state", PolicyFeature(type=FeatureType.STATE, shape=(self.max_state_dim,)),
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list[int]:
        return [0]

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self):
        return None
