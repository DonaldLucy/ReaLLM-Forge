from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn


_ACTIVE_PIGGYBACK_CONTEXT: Optional["PiggybackContext"] = None


def _sanitize_target_name(name: str) -> str:
    return name.replace(".", "__")


class GradientSideNet(nn.Module):
    """Two-layer element-wise MLP that predicts a residual correction to the gradient."""

    def __init__(self, hidden_dim: int = 32, input_dim: int = 6):
        super().__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, grad: torch.Tensor, quant_features: torch.Tensor) -> torch.Tensor:
        grad_dtype = grad.dtype
        grad_shape = grad.shape
        feature_list = [
            grad,
            grad.abs(),
            quant_features[..., 0],
            quant_features[..., 1],
            quant_features[..., 2],
            quant_features[..., 3],
        ]
        x = torch.stack(feature_list, dim=-1).reshape(-1, self.input_dim).to(self.fc1.weight.dtype)
        delta = self.fc2(self.act(self.fc1(x)))
        corrected = grad.reshape(-1, 1).to(delta.dtype) + delta
        return corrected.reshape(grad_shape).to(grad_dtype)


class SideRehearsalCollection(nn.Module):
    """Container attached to the main model so DDP can synchronize side-network params."""

    def __init__(self, target_names: Iterable[str], hidden_dim: int = 32):
        super().__init__()
        self._name_to_key: Dict[str, str] = {}
        self.nets = nn.ModuleDict()
        for target_name in target_names:
            key = _sanitize_target_name(target_name)
            self._name_to_key[target_name] = key
            self.nets[key] = GradientSideNet(hidden_dim=hidden_dim)

    def has_target(self, target_name: str) -> bool:
        return target_name in self._name_to_key

    def get_net(self, target_name: str) -> GradientSideNet:
        return self.nets[self._name_to_key[target_name]]

    def target_names(self) -> list[str]:
        return list(self._name_to_key.keys())


@dataclass
class PendingRehearsalState:
    target_name: str
    weight_before_step: torch.Tensor
    gradient: torch.Tensor
    learning_rate: float
    beta1: float
    beta2: float
    eps: float
    weight_decay: float
    adamw_step: int
    exp_avg: torch.Tensor
    exp_avg_sq: torch.Tensor
    quant_features: torch.Tensor
    source_iter: int


class PiggybackContext:
    """Runtime context for evaluating one pending candidate update on the next batch."""

    def __init__(
        self,
        *,
        target_name: str,
        side_net: GradientSideNet,
        pending_state: PendingRehearsalState,
        piggyback_k: int,
        base_batch_size: int,
    ):
        self.target_name = target_name
        self.side_net = side_net
        self.pending_state = pending_state
        self.piggyback_k = int(max(0, piggyback_k))
        self.base_batch_size = int(base_batch_size)

        self.injected = False
        self.injected_k = 0
        self.candidate_weight: Optional[torch.Tensor] = None
        self.candidate_exp_avg: Optional[torch.Tensor] = None
        self.candidate_exp_avg_sq: Optional[torch.Tensor] = None
        self.candidate_adamw_step: Optional[int] = None
        self.base_probe_loss: Optional[torch.Tensor] = None
        self.candidate_probe_loss: Optional[torch.Tensor] = None

    @property
    def has_probe(self) -> bool:
        return self.injected and self.injected_k > 0

    def should_inject(self, module_path: Optional[str], batch_size: int) -> bool:
        return (
            module_path == self.target_name
            and not self.injected
            and self.piggyback_k > 0
            and batch_size > 0
        )

    def build_candidate_weight(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.candidate_weight is None:
            grad = self.pending_state.gradient.to(device=device, dtype=dtype)
            base_weight = self.pending_state.weight_before_step.to(device=device, dtype=dtype)
            exp_avg = self.pending_state.exp_avg.to(device=device, dtype=dtype)
            exp_avg_sq = self.pending_state.exp_avg_sq.to(device=device, dtype=dtype)
            quant_features = self.pending_state.quant_features.to(device=device, dtype=dtype)
            corrected_grad = self.side_net(grad, quant_features)
            next_step = self.pending_state.adamw_step + 1

            exp_avg = exp_avg.mul(self.pending_state.beta1).add(
                corrected_grad,
                alpha=(1.0 - self.pending_state.beta1),
            )
            exp_avg_sq = exp_avg_sq.mul(self.pending_state.beta2).addcmul(
                corrected_grad,
                corrected_grad,
                value=(1.0 - self.pending_state.beta2),
            )

            bias_correction1 = 1.0 - self.pending_state.beta1 ** next_step
            bias_correction2 = 1.0 - self.pending_state.beta2 ** next_step

            decayed_weight = base_weight.mul(1.0 - self.pending_state.learning_rate * self.pending_state.weight_decay)
            denom = exp_avg_sq.sqrt().div(math.sqrt(bias_correction2)).add(self.pending_state.eps)
            step_size = self.pending_state.learning_rate / bias_correction1
            self.candidate_weight = decayed_weight.addcdiv(exp_avg, denom, value=-step_size)
            self.candidate_exp_avg = exp_avg
            self.candidate_exp_avg_sq = exp_avg_sq
            self.candidate_adamw_step = next_step
        return self.candidate_weight

    def mark_injected(self, actual_k: int) -> None:
        self.injected = True
        self.injected_k = int(actual_k)

    def set_probe_losses(self, *, base_probe_loss: torch.Tensor, candidate_probe_loss: torch.Tensor) -> None:
        self.base_probe_loss = base_probe_loss
        self.candidate_probe_loss = candidate_probe_loss

    def improvement(self) -> Optional[torch.Tensor]:
        if self.base_probe_loss is None or self.candidate_probe_loss is None:
            return None
        return self.base_probe_loss - self.candidate_probe_loss


@contextmanager
def activate_piggyback_context(ctx: Optional[PiggybackContext]):
    global _ACTIVE_PIGGYBACK_CONTEXT
    previous = _ACTIVE_PIGGYBACK_CONTEXT
    _ACTIVE_PIGGYBACK_CONTEXT = ctx
    try:
        yield
    finally:
        _ACTIVE_PIGGYBACK_CONTEXT = previous


def get_active_piggyback_context() -> Optional[PiggybackContext]:
    return _ACTIVE_PIGGYBACK_CONTEXT
