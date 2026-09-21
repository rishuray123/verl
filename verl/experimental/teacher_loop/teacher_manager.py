# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import asyncio
from typing import Any, Optional
from uuid import uuid4

import ray
import torch
from omegaconf import DictConfig
from torch.nn import functional as F

from verl.experimental.agent_loop import AsyncLLMServerManager
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import (
    DistillationConfig,
    DistillationLossConfig,
    DistillationTeacherModelConfig,
)


def _get_teacher_sampling_params(
    teacher_model_config: DistillationTeacherModelConfig,
    distillation_loss_config: DistillationLossConfig,
    mix_teachers: bool = False,
) -> dict[str, Any]:
    """Get sampling parameters for teacher model when computing log probabilities for distillation."""
    if teacher_model_config.inference.temperature != 1.0:
        raise NotImplementedError("vLLM does not support temperature for prompt_logprobs.")

    use_topk = distillation_loss_config.loss_settings.use_topk or mix_teachers
    num_logprobs = distillation_loss_config.topk if use_topk else 0
    if mix_teachers:
        num_logprobs = max(int(num_logprobs or 0), int(distillation_loss_config.topk or 8))
    return {
        "max_tokens": 1,
        "temperature": teacher_model_config.inference.temperature,
        "prompt_logprobs": num_logprobs,
    }


def _dense_topk(ids_rows, logprob_rows, floor: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack possibly-sparse vLLM top-k rows into [S, K] tensors (missing slots = floor / -1)."""
    k = max(len(row) for row in logprob_rows)
    seqlen = len(logprob_rows)
    ids = torch.full((seqlen, k), -1, dtype=torch.int64)
    logp = torch.full((seqlen, k), floor, dtype=torch.float32)
    for t, (id_row, lp_row) in enumerate(zip(ids_rows, logprob_rows)):
        for j, (token_id, lp) in enumerate(zip(id_row, lp_row)):
            if token_id is None or lp is None:
                continue
            ids[t, j] = int(token_id)
            logp[t, j] = float(lp)
    return ids, logp


def _lookup_token_logprob(teacher_ids: torch.Tensor, teacher_logprobs: torch.Tensor, y: torch.Tensor, floor: float):
    """log p(y_t) from top-k tables, or the scalar column when K=1."""
    if teacher_logprobs.ndim == 1:
        n = min(teacher_logprobs.numel(), y.numel())
        return teacher_logprobs[:n]
    n = min(teacher_logprobs.shape[0], y.numel())
    ids = teacher_ids[:n]
    logp = teacher_logprobs[:n]
    y = y[:n]
    if ids.ndim == 1:
        ids = ids.unsqueeze(-1)
        logp = logp.unsqueeze(-1)
    match = ids == y.unsqueeze(-1)
    has = match.any(dim=-1)
    col = match.long().argmax(dim=-1)
    gathered = logp.gather(1, col.unsqueeze(-1)).squeeze(-1)
    return torch.where(has, gathered, torch.full_like(gathered, floor))


def _topk_entropy(teacher_logprobs: torch.Tensor) -> torch.Tensor:
    logp = teacher_logprobs.float()
    if logp.ndim == 1:
        return torch.zeros_like(logp)
    probs = torch.softmax(logp, dim=-1)
    return -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)


def mix_teacher_token_logprobs(
    vl_ids: torch.Tensor,
    vl_logprobs: torch.Tensor,
    text_ids: torch.Tensor,
    text_logprobs: torch.Tensor,
    sequence_ids: list[int],
    temperature: float = 1.0,
    floor: float = -10.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Entropy-weighted mix of two teachers' p(y_t). Low entropy → higher weight."""
    y = torch.tensor(sequence_ids, dtype=torch.int64)
    tau = max(float(temperature), 1e-6)
    log_vl = _lookup_token_logprob(vl_ids, vl_logprobs, y, floor)
    log_tx = _lookup_token_logprob(text_ids, text_logprobs, y, floor)
    h_vl = _topk_entropy(vl_logprobs)
    h_tx = _topk_entropy(text_logprobs)
    n = min(y.numel(), log_vl.numel(), log_tx.numel(), h_vl.numel(), h_tx.numel())
    log_vl, log_tx = log_vl[:n], log_tx[:n]
    h_vl, h_tx = h_vl[:n], h_tx[:n]
    w_vl = torch.exp(-h_vl / tau)
    w_tx = torch.exp(-h_tx / tau)
    alpha = w_vl / (w_vl + w_tx).clamp_min(1e-12)
    mixed_p = alpha * log_vl.exp() + (1.0 - alpha) * log_tx.exp()
    mixed_logp = mixed_p.clamp_min(1e-12).log()
    stats = {
        "alpha_vl": float(alpha.mean().item()),
        "H_vl": float(h_vl.mean().item()),
        "H_text": float(h_tx.mean().item()),
    }
    return mixed_logp, stats


def _pad_teacher_outputs(
    teacher_ids: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    prompt_width: int,
    response_width: int,
    prompt_length: int,
    response_length: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # TODO(wuxibin): remove padding and use tensordict.
    left_pad_size = prompt_width - prompt_length
    right_pad_size = response_width - response_length
    padding = (0, 0, left_pad_size, right_pad_size)
    return (
        F.pad(teacher_ids, padding, value=pad_token_id).unsqueeze(0),
        F.pad(teacher_logprobs, padding, value=0.0).unsqueeze(0),
    )


class AsyncTeacherLLMServerManager:
    """Teacher-specific async client used for distillation logprob computation."""

    def __init__(
        self,
        config: DictConfig,
        servers: dict[str, list[tuple[str, ray.actor.ActorHandle]]],
        load_balancer_handle: dict[str, ray.actor.ActorHandle],
    ):
        self.distillation_config: DistillationConfig = omega_conf_to_dataclass(config.distillation)
        self.distillation_loss_config: DistillationLossConfig = self.distillation_config.distillation_loss
        self.teacher_key: str = self.distillation_config.teacher_key

        self.teacher_model_configs: dict[str, DistillationTeacherModelConfig] = self.distillation_config.teacher_models
        expected = set(self.teacher_model_configs)
        if set(servers.keys()) != expected:
            raise ValueError(f"server keys {sorted(servers)} do not match teacher routing keys {sorted(expected)}.")
        if set(load_balancer_handle.keys()) != expected:
            raise ValueError(
                f"load_balancer_handle keys {sorted(load_balancer_handle)} do not match "
                f"teacher routing keys {sorted(expected)}."
            )

        self.server_managers: dict[str, AsyncLLMServerManager] = {
            key: AsyncLLMServerManager(
                config=config,
                servers=servers[key],
                load_balancer_handle=load_balancer_handle[key],
            )
            for key in self.teacher_model_configs
        }

    def _resolve_teacher_key(self, routing_key: Optional[str]) -> str:
        if len(self.teacher_model_configs) == 1:
            # Single-teacher path: route everything to the one teacher regardless of the sample's key.
            return next(iter(self.teacher_model_configs))
        if routing_key is None:
            raise ValueError(
                f"Routing key is required for multi-teacher distillation "
                f"(configured via distillation.teacher_key={self.teacher_key!r})."
            )
        if routing_key not in self.teacher_model_configs:
            raise ValueError(
                f"No teacher configured for routing key {routing_key!r}. "
                f"Configured teachers: {sorted(self.teacher_model_configs)}."
            )
        return routing_key

    def _vl_text_keys(self) -> tuple[str, str]:
        keys = list(self.teacher_model_configs)
        vl = next((k for k in keys if "vl" in k.lower()), None)
        text = next((k for k in keys if "text" in k.lower() or "lm" in k.lower()), None)
        if vl is None or text is None or vl == text:
            if len(keys) != 2:
                raise ValueError(
                    f"mix_teachers needs a VL and a text teacher; configured keys: {sorted(keys)}"
                )
            vl, text = keys[0], keys[1]
        return vl, text

    async def _score_teacher(
        self,
        teacher_key: str,
        sequence_ids: list[int],
        image_data=None,
        video_data=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        teacher_model_config = self.teacher_model_configs[teacher_key]
        server_manager = self.server_managers[teacher_key]
        mix = bool(getattr(self.distillation_config, "mix_teachers", False))
        teacher_output = await server_manager.generate(
            request_id=uuid4().hex,
            prompt_ids=sequence_ids,
            sampling_params=_get_teacher_sampling_params(
                teacher_model_config, self.distillation_loss_config, mix_teachers=mix
            ),
            image_data=image_data,
            video_data=video_data,
        )
        floor = float(self.distillation_loss_config.log_prob_min_clamp or -10.0)
        ids, logp = _dense_topk(
            teacher_output.extra_fields["prompt_ids"],
            teacher_output.extra_fields["prompt_logprobs"],
            floor,
        )
        if ids.shape[0] != len(sequence_ids):
            raise AssertionError(
                f"{teacher_key} prompt_logprobs length {ids.shape[0]} != sequence {len(sequence_ids)}"
            )
        return ids, logp

    async def compute_teacher_logprobs_single(
        self,
        sequence_ids: list[int],
        multi_modal_data: Optional[dict[str, Any]] = None,
        routing_key: Optional[str] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute teacher log probabilities for a single unpadded sequence."""
        multi_modal_data = multi_modal_data or {}
        self.last_mix_stats: dict[str, float] = {}
        mix = bool(getattr(self.distillation_config, "mix_teachers", False))
        if mix and len(self.teacher_model_configs) > 1:
            vl_key, text_key = self._vl_text_keys()
            vl_result, text_result = await asyncio.gather(
                self._score_teacher(
                    vl_key,
                    sequence_ids,
                    image_data=multi_modal_data.get("images"),
                    video_data=multi_modal_data.get("videos"),
                ),
                self._score_teacher(text_key, sequence_ids, image_data=None, video_data=None),
                return_exceptions=True,
            )
            floor = float(self.distillation_loss_config.log_prob_min_clamp or -10.0)
            tau = float(getattr(self.distillation_config, "mix_temperature", 1.0) or 1.0)
            y = torch.tensor(sequence_ids, dtype=torch.int64)

            def _as_sampled(ids, logp):
                sampled = _lookup_token_logprob(ids, logp, y, floor)
                return y.to(torch.int32).unsqueeze(-1), sampled.unsqueeze(-1)

            if isinstance(vl_result, Exception) and isinstance(text_result, Exception):
                raise vl_result
            if isinstance(vl_result, Exception):
                teacher_ids, teacher_logprobs = _as_sampled(*text_result)
                self.last_mix_stats = {"alpha_vl": 0.0, "H_vl": float("nan"), "H_text": 0.0}
            elif isinstance(text_result, Exception):
                teacher_ids, teacher_logprobs = _as_sampled(*vl_result)
                self.last_mix_stats = {"alpha_vl": 1.0, "H_vl": 0.0, "H_text": float("nan")}
            else:
                mixed, stats = mix_teacher_token_logprobs(
                    vl_result[0],
                    vl_result[1],
                    text_result[0],
                    text_result[1],
                    sequence_ids,
                    temperature=tau,
                    floor=floor,
                )
                teacher_ids = y.to(torch.int32).unsqueeze(-1)
                teacher_logprobs = mixed.unsqueeze(-1)
                self.last_mix_stats = stats
            return teacher_ids, teacher_logprobs

        teacher_key = self._resolve_teacher_key(routing_key)
        use_images = "vl" in teacher_key.lower()
        teacher_ids, teacher_logprobs = await self._score_teacher(
            teacher_key,
            sequence_ids,
            image_data=multi_modal_data.get("images") if use_images else None,
            video_data=multi_modal_data.get("videos") if use_images else None,
        )
        return teacher_ids, teacher_logprobs
