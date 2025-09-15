# Copyright 2025 The HuggingFace Inc. team. All rights reserved.

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig


@PreTrainedConfig.register_subclass("internvla")
@dataclass
class InternVLAConfig(PreTrainedConfig):
    # IO shape
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # State/action padding upper bounds
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Image preprocessing
    resize_imgs_with_padding: tuple[int, int] = (448, 448)
    empty_cameras: int = 0

    # Language
    tokenizer_max_length: int = 64
    pad_language_to: str = "longest"  # or "max_length"

    # Flow-matching sampling
    num_steps: int = 10
    min_period: float = 4e-3
    max_period: float = 4.0

    # Attention / caching
    use_cache: bool = True
    attention_mode: str = "cross_attn"  # "self_attn" | "cross_attn"
    self_attn_every_n_layers: int = 2
    
    # Knowledge insulation (pi0.5-style)
    knowledge_insulation: bool = True

    # Finetuning / freezing
    freeze_vision_encoder: bool = True
    # Default: enable VLM training (insulated via KI); Expert-only can be re-enabled via CLI
    train_expert_only: bool = False if knowledge_insulation else True
    train_state_proj: bool = True

    # Expert/VLM sizing controls
    vlm_model_name: str = "OpenGVLab/InternVL3_5-4B-HF"
    load_vlm_weights: bool = True
    # Size down Expert to ~500M by default (adjust as needed)
    num_expert_layers: int = 12   # VLM depth divisor (e.g., 36→12)
    num_vlm_layers: int = 12      # Leave VLM depth unchanged by default
    expert_width_multiplier: float = 0.5

    # Discrete auxiliary (FAST) for KI co-training
    use_discrete_aux: bool = True if knowledge_insulation else False
    discrete_loss_weight: float = 0.5
    fast_repo_id: str = "physical-intelligence/fast"
    fast_skip_tokens: int = 128

    # Training presets
    optimizer_lr: float = 5e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10.0
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 50_000
    scheduler_decay_lr: float = 5e-5

    def __post_init__(self):
        super().__post_init__()
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) must be <= chunk_size ({self.chunk_size})."
            )

    def validate_features(self) -> None:
        # Optionally add empty cameras
        for i in range(self.empty_cameras):
            key = f"observation.images.empty_camera_{i}"
            self.input_features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640))
        # Auto-shrink state/action pad dims to dataset shapes to reduce projection params
        try:
            rs = self.robot_state_feature
            if rs is not None and len(rs.shape) > 0:
                dim = int(rs.shape[0])
                if dim > 0 and dim <= self.max_state_dim:
                    self.max_state_dim = dim
        except Exception:
            pass
        try:
            af = self.action_feature
            if af is not None and len(af.shape) > 0:
                dim = int(af.shape[0])
                if dim > 0 and dim <= self.max_action_dim:
                    self.max_action_dim = dim
        except Exception:
            pass

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

    @property
    def observation_delta_indices(self) -> list:
        return [0]

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
