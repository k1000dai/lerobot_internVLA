from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig


@PreTrainedConfig.register_subclass("internvl35")
@dataclass
class InternVL35Config(PreTrainedConfig):
    """Configuration for the InternVL3.5 based VLA.

    The structure largely follows :class:`SmolVLAConfig` but targets the
    `OpenGVLab/InternVL3_5-4B-HF` vision-language backbone.  The action expert is
    trained with flow matching and the vision-language model weights are frozen
    by default ("knowledge insulation").
    """

    # Input / output structure
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

    max_state_dim: int = 32
    max_action_dim: int = 32

    resize_imgs_with_padding: tuple[int, int] = (512, 512)
    empty_cameras: int = 0

    tokenizer_max_length: int = 48
    num_steps: int = 10

    use_cache: bool = True

    # Training utilities
    freeze_vlm: bool = True  # Knowledge insulation
    flow_matching: bool = True

    vlm_model_name: str = "OpenGVLab/InternVL3_5-4B-HF"
    add_image_special_tokens: bool = False
    attention_mode: str = "cross_attn"

    num_vlm_layers: int = -1
    num_expert_layers: int = -1
    expert_width_multiplier: float = 1.0

    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10.0

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    def validate_features(self) -> None:  # pragma: no cover - simple passthrough
        for i in range(self.empty_cameras):
            key = f"observation.images.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640),
            )
            self.input_features[key] = empty_camera

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
