#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.

import math
from collections import deque

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.constants import ACTION, OBS_STATE
from lerobot.policies.normalize import Normalize, Unnormalize
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import populate_queues
from lerobot.utils.utils import get_safe_dtype

from .configuration_internvla import InternVLAConfig
from .internvl_with_expert import InternVLWithExpertModel


def create_sinusoidal_pos_embedding(
    time: torch.Tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size,)`.")
    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def make_att_2d_masks(pad_masks: torch.Tensor, att_masks: torch.Tensor) -> torch.Tensor:
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def resize_with_pad(img: torch.Tensor, width: int, height: int, pad_value: float = 0.0) -> torch.Tensor:
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")
    cur_h, cur_w = img.shape[2:]
    ratio = max(cur_w / width, cur_h / height)
    rh, rw = int(cur_h / ratio), int(cur_w / ratio)
    resized = F.interpolate(img, size=(rh, rw), mode="bilinear", align_corners=False)
    ph, pw = max(0, int(height - rh)), max(0, int(width - rw))
    return F.pad(resized, (pw, 0, ph, 0), value=pad_value)


def pad_vector(x: torch.Tensor, new_dim: int) -> torch.Tensor:
    if x.shape[-1] == new_dim:
        return x
    shape = list(x.shape)
    cur = shape[-1]
    shape[-1] = new_dim
    out = torch.zeros(*shape, dtype=x.dtype, device=x.device)
    out[..., :cur] = x
    return out


class InternVLAPolicy(PreTrainedPolicy):
    config_class = InternVLAConfig
    name = "internvla"

    def __init__(self, config: InternVLAConfig, dataset_stats: dict[str, dict[str, Tensor]] | None = None):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(config.output_features, config.normalization_mapping, dataset_stats)
        self.unnormalize_outputs = Unnormalize(config.output_features, config.normalization_mapping, dataset_stats)

        self.model = InternVLAFlowMatching(config)

        self.reset()

    def get_optim_params(self):
        return self.parameters()

    def reset(self):
        self._queues: dict[str, deque] = {ACTION: deque([], maxlen=self.config.n_action_steps)}

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        self.eval()
        batch = self.normalize_inputs(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])
        actions = self._get_action_chunk(batch, noise)
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        self.eval()
        batch = self.normalize_inputs(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])
        if len(self._queues[ACTION]) == 0:
            actions = self._get_action_chunk(batch, noise)
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])
        return self._queues[ACTION].popleft()

    def _get_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None) -> Tensor:
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)
        actions = self.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=noise)

        # Remove padding and unnormalize
        actions = actions[:, :, : self.config.action_feature.shape[0]]
        actions = self.unnormalize_outputs({ACTION: actions})[ACTION]
        return actions

    def forward(self, batch: dict[str, Tensor], noise=None, time=None) -> tuple[Tensor, dict]:
        # Keep raw elements needed for auxiliary discrete branch
        raw_actions = batch.get(ACTION, None)
        raw_tasks = batch.get("task", None)

        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        losses = self.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time)
        loss = losses.mean()

        # Optional discrete auxiliary loss (FAST on-the-fly). Now conditions on images + text state
        if self.config.use_discrete_aux and raw_actions is not None:
            aux_loss = self._compute_discrete_aux_loss(raw_actions, raw_tasks, images=images, state=state)
            loss = loss + self.config.discrete_loss_weight * aux_loss

        return loss, {"loss": loss.item()}

    def _compute_discrete_aux_loss(self, raw_actions: Tensor, tasks, *, images=None, state: Tensor | None = None) -> Tensor:
        """Compute CE loss on FAST-discretized actions with the VLM text head.

        Uses pixel images (first available camera) and a discretized text-state in the prefix,
        matching the intended VLM conditioning (image encoder + prompt + text state).
        """
        device = raw_actions.device
        bsz = raw_actions.shape[0]

        # Lazy-load processors/tokenizers
        fast_proc = getattr(self, "_fast_processor", None)
        if fast_proc is None:
            from transformers import AutoProcessor  # type: ignore

            self._fast_processor = AutoProcessor.from_pretrained(
                self.config.fast_repo_id, trust_remote_code=True
            )
            fast_proc = self._fast_processor

        vlm_tok = None
        proc = getattr(self.model.vlm_with_expert, "processor", None)
        if proc is not None and hasattr(proc, "tokenizer"):
            vlm_tok = proc.tokenizer
        if vlm_tok is None:
            from transformers import AutoTokenizer  # type: ignore

            vlm_tok = AutoTokenizer.from_pretrained(self.config.vlm_model_name)

        # Prepare texts (use the same prompt as Expert: raw task only)
        if tasks is None:
            tasks = [""]
        if isinstance(tasks, str):
            tasks = [tasks]
        if len(tasks) == 1:
            tasks = [tasks[0] for _ in range(bsz)]

        # Normalize actions to [-1,1] per sample/time dim as in PI0FAST
        def _minmax_norm(x: torch.Tensor) -> torch.Tensor:
            mins = x.amin(dim=(1, 2), keepdim=True)
            maxs = x.amax(dim=(1, 2), keepdim=True)
            return 2 * (x - mins) / (maxs - mins + 1e-8) - 1

        act = raw_actions
        if act.ndim == 2:
            act = act[:, None, :]
        # Pad to configured action dim
        act = pad_vector(act, self.config.max_action_dim)
        act_norm = _minmax_norm(act)

        # FAST encode batch
        fast_out = fast_proc(act_norm.cpu())
        # physical-intelligence/fast returns List[List[int]] (or dict with input_ids)
        if isinstance(fast_out, dict):
            fast_tokens = fast_out.get("input_ids", None)
            if fast_tokens is None:
                raise TypeError("FAST processor did not return 'input_ids'.")
        elif isinstance(fast_out, (list, tuple)):
            fast_tokens = fast_out
        else:
            raise TypeError(f"Unsupported FAST output type: {type(fast_out)}")

        # Prefix texts = tasks only (no explicit 'Task:' prefix and no text-state; match Expert branch)
        prefix_texts = [(tasks[i] if isinstance(tasks[i], str) else "").strip() for i in range(bsz)]
        # Tokenize prefix to get VLM token IDs
        pref = vlm_tok(prefix_texts, add_special_tokens=True, padding=False, return_tensors=None)

        # Map FAST token ids into the tail of the VLM vocab (like PI0-FAST)
        vocab_size = getattr(vlm_tok, "vocab_size", None)
        if vocab_size is None:
            raise RuntimeError("VLM tokenizer does not expose vocab_size")
        skip = getattr(self.config, "fast_skip_tokens", 128)

        def map_fast_to_vlm_ids(seq: list[int]) -> list[int]:
            return [int(max(0, vocab_size - 1 - skip - x)) for x in seq]

        # Build per-sample concatenated inputs and labels
        concat_ids, attention_mask, labels = [], [], []
        bos = vlm_tok("Action: ", add_special_tokens=False, return_tensors=None)
        bos_ids = bos["input_ids"][0] if isinstance(bos["input_ids"], list) else bos["input_ids"].tolist()[0]
        eos_id = vlm_tok.eos_token_id if hasattr(vlm_tok, "eos_token_id") else None

        for i in range(bsz):
            ids_pref = pref["input_ids"][i]
            ids_act = map_fast_to_vlm_ids(list(fast_tokens[i]))
            ids = ids_pref + bos_ids + ids_act + ([eos_id] if eos_id is not None else [])
            mask = [1] * len(ids)
            lab = [-100] * (len(ids_pref) + len(bos_ids)) + ids_act[:] + ([-100] if eos_id is not None else [])
            concat_ids.append(ids)
            attention_mask.append(mask)
            labels.append(lab)

        # Pad to tensor
        batch_inputs = {"input_ids": concat_ids, "attention_mask": attention_mask}
        padded = vlm_tok.pad(batch_inputs, padding=True, return_tensors="pt")
        input_ids = padded["input_ids"].to(device)
        attn = padded["attention_mask"].to(device)

        # Align labels shape to padding
        max_len = input_ids.shape[1]
        lab_padded = torch.full((bsz, max_len), -100, dtype=torch.long)
        for i, lab in enumerate(labels):
            lab_padded[i, : len(lab)] = torch.tensor(lab, dtype=torch.long)
        lab_padded = lab_padded.to(device)

        # Prepare pixel values (use all available cameras, same as Expert)
        pv = None
        if images is not None and isinstance(images, list) and len(images) > 0:
            try:
                vt_param = next(self.model.vlm_with_expert.get_vlm_model().vision_tower.parameters())
                vt_device = vt_param.device
            except Exception:
                vt_device = input_ids.device
            pvs = []
            for img in images:
                if img is None:
                    continue
                pvs.append(img.to(device=vt_device, dtype=torch.bfloat16))
            if len(pvs) == 1:
                pv = pvs[0]
            elif len(pvs) > 1:
                # Many HF VLMs accept a list of pixel tensors for multi-image conditioning
                pv = pvs

        # Compute CE loss with the VLM head (conditioned on images + text prompt)
        if pv is not None:
            outputs = self.model.vlm_with_expert.vlm(
                pixel_values=pv, input_ids=input_ids, attention_mask=attn, labels=lab_padded, use_cache=False
            )
        else:
            outputs = self.model.vlm_with_expert.vlm(
                input_ids=input_ids, attention_mask=attn, labels=lab_padded, use_cache=False
            )
        return outputs.loss

    def prepare_images(self, batch: dict[str, Tensor]):
        images = []
        img_masks = []
        present = [k for k in self.config.image_features if k in batch]
        missing = [k for k in self.config.image_features if k not in batch]
        if len(present) == 0:
            raise ValueError("At least one image feature expected in batch for InternVLA.")
        for key in present:
            img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]
            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0.0)
            bsz = img.shape[0]
            device = img.device
            mask = batch.get(f"{key}_padding_mask", torch.ones(bsz, dtype=torch.bool, device=device)).bool()
            images.append(img)
            img_masks.append(mask)
        for n in range(len(missing)):
            if n >= self.config.empty_cameras:
                break
            images.append(torch.zeros_like(images[-1]))
            img_masks.append(torch.zeros_like(img_masks[-1]))
        return images, img_masks

    def prepare_language(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        device = batch[OBS_STATE].device
        tasks = batch.get("task", None)
        if tasks is None:
            # Fallback: empty instruction
            tasks = [""]
        if isinstance(tasks, str):
            tasks = [tasks]
        if len(tasks) == 1:
            tasks = [tasks[0] for _ in range(batch[OBS_STATE].shape[0])]
        # Basic tokenization via VLM processor tokenizer if available
        tokenizer = getattr(self.model.vlm_with_expert, "processor", None)
        if tokenizer is not None and hasattr(tokenizer, "tokenizer"):
            t = tokenizer.tokenizer(
                tasks,
                padding=self.config.pad_language_to,
                max_length=self.config.tokenizer_max_length,
                truncation=True,
                return_tensors="pt",
            )
        else:
            # Very conservative fallback (no truncation/padding)
            from transformers import AutoTokenizer  # pragma: no cover

            tk = AutoTokenizer.from_pretrained(self.config.vlm_model_name)
            t = tk(tasks, padding=self.config.pad_language_to, return_tensors="pt")
        lang_tokens = t["input_ids"].to(device=device)
        lang_masks = t["attention_mask"].to(device=device, dtype=torch.bool)
        return lang_tokens, lang_masks

    def prepare_state(self, batch: dict[str, Tensor]) -> torch.Tensor:
        state = batch[OBS_STATE][:, -1, :] if batch[OBS_STATE].ndim > 2 else batch[OBS_STATE]
        state = pad_vector(state, self.config.max_state_dim)
        return state


class InternVLAFlowMatching(nn.Module):
    """
    InternVLA: InternVL 3.5 4B + Action Expert, trained with Flow Matching (OpenPI-style).
    """

    def __init__(self, config: InternVLAConfig):
        super().__init__()
        self.config = config

        self.vlm_with_expert = InternVLWithExpertModel(
            model_id=self.config.vlm_model_name,
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            load_vlm_weights=self.config.load_vlm_weights,
            attention_mode=self.config.attention_mode,
            num_expert_layers=self.config.num_expert_layers,
            num_vlm_layers=self.config.num_vlm_layers,
            self_attn_every_n_layers=self.config.self_attn_every_n_layers,
            expert_width_multiplier=self.config.expert_width_multiplier,
            knowledge_insulation=self.config.knowledge_insulation,
        )

        # Projections
        self.state_proj = nn.Linear(self.config.max_state_dim, self.vlm_with_expert.vlm_hidden_size)
        # Action expert width matches expert
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.vlm_with_expert.expert_hidden_size)
        self.action_out_proj = nn.Linear(self.vlm_with_expert.expert_hidden_size, self.config.max_action_dim)

        # Time fusion for base (no-knowledge-insulation)
        if not self.config.knowledge_insulation:
            self.action_time_mlp_in = nn.Linear(
                2 * self.vlm_with_expert.expert_hidden_size, self.vlm_with_expert.expert_hidden_size
            )
            self.action_time_mlp_out = nn.Linear(
                self.vlm_with_expert.expert_hidden_size, self.vlm_with_expert.expert_hidden_size
            )
        else:
            # Knowledge insulation: derive adarms condition from time only, per-step
            self.time_mlp_in = nn.Linear(self.vlm_with_expert.expert_hidden_size, self.vlm_with_expert.expert_hidden_size)
            self.time_mlp_out = nn.Linear(self.vlm_with_expert.expert_hidden_size, self.vlm_with_expert.expert_hidden_size)

    # Sampling helpers
    def sample_noise(self, shape, device):
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)

    def sample_time(self, bsize, device):
        # Beta(1.5,1) in (0.001,0.999), float32
        alpha, beta = torch.tensor(1.5, device=device), torch.tensor(1.0, device=device)
        t = torch.distributions.Beta(alpha, beta).sample((bsize,))
        return (t * 0.999 + 0.001).to(dtype=torch.float32)

    # Embedding builders
    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks, state):
        embs = []
        pad_masks = []
        att_masks = []
        # Images
        for img, mask in zip(images, img_masks, strict=False):
            img_emb = self.vlm_with_expert.embed_image(img.to(dtype=torch.bfloat16))
            # normalize as in eager impl
            img_emb = img_emb * math.sqrt(img_emb.shape[-1])
            if self.config.knowledge_insulation:
                img_emb = img_emb.detach()
            bsz, n = img_emb.shape[:2]
            img_mask = mask[:, None].expand(bsz, n)
            embs.append(img_emb)
            pad_masks.append(img_mask)
            att_masks += [0] * n
        # Language
        lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
        if self.config.knowledge_insulation:
            lang_emb = lang_emb.detach()
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_masks += [0] * lang_emb.shape[1]
        # State token
        st = self.state_proj(state)
        # Match dtype to previous embeddings (usually VLM bfloat16)
        if len(embs) > 0:
            st = st.to(dtype=embs[-1].dtype)
        st = st[:, None, :]
        embs.append(st)
        bsz = st.shape[0]
        device = st.device
        st_mask = torch.ones(bsz, 1, dtype=torch.bool, device=device)
        pad_masks.append(st_mask)
        # State (and later action) are not attended by prefix tokens
        att_masks += [1]

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        # Use integer mask for cumulative attention construction (avoid bool cumsum)
        att_masks = torch.tensor(att_masks, dtype=torch.int32, device=pad_masks.device)[None, :].expand(bsz, -1)
        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions, timestep):
        embs = []
        pad_masks = []
        att_masks = []

        act_emb = self.action_in_proj(noisy_actions)
        device = act_emb.device
        bsz = act_emb.shape[0]
        dtype = act_emb.dtype

        # Time embedding
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.vlm_with_expert.expert_hidden_size, self.config.min_period, self.config.max_period, device
        )
        time_emb = time_emb.type(dtype=dtype)

        adarms_cond = None
        if self.config.knowledge_insulation:
            # Per-step condition; match action horizon with expand
            te = self.time_mlp_in(time_emb)
            te = F.silu(te)
            te = self.time_mlp_out(te)
            te = F.silu(te)
            adarms_cond = te  # (B, E)
            emb = act_emb  # no fusion here; expert will receive time cond via FiLM
        else:
            te = time_emb[:, None, :].expand_as(act_emb)
            emb = torch.cat([act_emb, te], dim=2)
            emb = self.action_time_mlp_in(emb)
            emb = F.silu(emb)
            emb = self.action_time_mlp_out(emb)

        embs.append(emb)
        pad_masks.append(torch.ones(bsz, emb.shape[1], dtype=torch.bool, device=device))
        att_masks += [1] + ([0] * (emb.shape[1] - 1))
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        # Integer mask for cumulative attention
        att_masks = torch.tensor(att_masks, dtype=torch.int32, device=embs.device)
        att_masks = att_masks[None, :].expand(bsz, len(att_masks))
        return embs, pad_masks, att_masks, adarms_cond

    # FM loss / training
    def forward(self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None) -> Tensor:
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)
        x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad, prefix_att = self.embed_prefix(images, img_masks, lang_tokens, lang_masks, state)
        suffix_embs, suffix_pad, suffix_att, adarms_cond = self.embed_suffix(x_t, time)

        pad = torch.cat([prefix_pad, suffix_pad], dim=1)
        att = torch.cat([prefix_att, suffix_att], dim=1)
        att_2d = make_att_2d_masks(pad, att)
        pos_ids = torch.cumsum(pad, dim=1) - 1

        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d,
            position_ids=pos_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
            adarms_cond=adarms_cond,
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        losses = F.mse_loss(u_t, v_t, reduction="none")
        return losses

    # Inference / sampling
    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None) -> Tensor:
        bsz, device = state.shape[0], state.device
        if noise is None:
            noise = self.sample_noise((bsz, self.config.chunk_size, self.config.max_action_dim), device)

        prefix_embs, prefix_pad, prefix_att = self.embed_prefix(images, img_masks, lang_tokens, lang_masks, state)
        prefix_att2d = make_att_2d_masks(prefix_pad, prefix_att)
        prefix_pos = torch.cumsum(prefix_pad, dim=1) - 1
        _, past_kv = self.vlm_with_expert.forward(
            attention_mask=prefix_att2d,
            position_ids=prefix_pos,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
            adarms_cond=None,
        )

        dt = torch.tensor(-1.0 / self.config.num_steps, dtype=torch.float32, device=device)
        x_t = noise
        t = torch.tensor(1.0, dtype=torch.float32, device=device)
        while t >= -dt / 2:
            v_t = self._denoise_step(prefix_pad, past_kv, x_t, t)
            x_t = x_t + dt * v_t
            t = t + dt
        return x_t

    def _denoise_step(self, prefix_pad, past_kv, x_t, timestep):
        suffix_embs, suffix_pad, suffix_att, adarms_cond = self.embed_suffix(x_t, timestep.expand(x_t.shape[0]))
        suff_len = suffix_pad.shape[1]
        bsz, pref_len = prefix_pad.shape[0], prefix_pad.shape[1]
        pref_pad2d = prefix_pad[:, None, :].expand(bsz, suff_len, pref_len)
        suff_att2d = make_att_2d_masks(suffix_pad, suffix_att)
        att_full = torch.cat([pref_pad2d, suff_att2d], dim=2)
        pref_offsets = torch.sum(prefix_pad, dim=-1)[:, None]
        pos_ids = pref_offsets + torch.cumsum(suffix_pad, dim=1) - 1

        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=att_full,
            position_ids=pos_ids,
            past_key_values=past_kv,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
            adarms_cond=adarms_cond,
        )
        suff_out = outputs_embeds[1][:, -self.config.chunk_size :]
        suff_out = suff_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suff_out)
        return v_t
