#!/usr/bin/env python

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoProcessor

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.internvl35.configuration_internvl35 import InternVL35Config


class FlowMatchingActionExpert(nn.Module):
    """A minimal flow-matching based action head.

    This module is inspired by the implementation in the `openpi` repository and
    is intended to learn a time-conditioned flow from latent features to action
    space.
    """

    def __init__(self, hidden_size: int, action_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.fc1(hidden))
        return self.fc2(x)


class KnowledgeInsulator:
    """Utility to freeze parameters of a module (knowledge insulation)."""

    def __init__(self, module: nn.Module):
        for p in module.parameters():
            p.requires_grad_(False)


class InternVL35Policy(PreTrainedPolicy):
    config_class = InternVL35Config
    name = "internvl35"

    def __init__(self, config: InternVL35Config):
        super().__init__(config)

        self.processor = AutoProcessor.from_pretrained(config.vlm_model_name)
        self.vlm = AutoModelForCausalLM.from_pretrained(config.vlm_model_name)

        hidden_size = self.vlm.config.hidden_size
        self.action_expert = FlowMatchingActionExpert(hidden_size, config.max_action_dim)

        if config.freeze_vlm:
            KnowledgeInsulator(self.vlm)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        images = batch["observation.images"]
        texts = batch.get("text", [""] * images.shape[0])

        inputs = self.processor(images=images, text=texts, return_tensors="pt").to(self.vlm.device)
        outputs = self.vlm(**inputs, use_cache=self.config.use_cache)
        hidden = outputs.last_hidden_state[:, -1]
        actions = self.action_expert(hidden)
        return {"actions": actions}
