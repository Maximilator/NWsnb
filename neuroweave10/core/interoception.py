"""Интероцепция: нейроны-наблюдатели, читающие состояние соседей.

Слой саморефлексии: N_int «интеро-нейронов» на каждую зону получают не
входной сигнал, а внутреннее состояние соседнего пула (средний V_S,
энергия ATP, Ca-нагрузка, локальная частота) и кодируют его в
собственную спайковую активность — материал для отчёта AST о
состоянии организма (метапознание: сеть знает, *в каком* она состоянии,
а не только *что* обрабатывает). Связь: интеро-активность идёт в AST
как признак и в GW как дополнительный кандидат — состояние самой сети
может стать содержимым сознания.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class InteroOut:
    state_vec: torch.Tensor      # [B,N_int] активность интеро-слоя
    summary: Dict[str, float]    # агрегированное само-состояние


class InteroceptiveField:
    """Интеро-нейроны: читают пул соседей (v_s, energy, ca, rate)."""

    def __init__(self, zone_names: list, n_int: int = 16, seed: int = 0):
        self.names = zone_names
        self.n_int = n_int
        g = torch.Generator().manual_seed(seed)
        self.w = torch.randn(n_int, 4, generator=g) * 0.8
        self.bias = torch.randn(n_int, generator=g) * 0.2

    def read(self, zone_states: Dict[str, torch.Tensor]) -> InteroOut:
        """zone_states[name] = [v_s_mean, energy_mean, ca_mean, rate]."""
        feats = torch.stack([zone_states[n] for n in self.names])  # [Z,4]
        act = torch.sigmoid(feats @ self.w.t() + self.bias)        # [Z,N_int]
        self.state_vec = act
        return InteroOut(act.flatten(),
                         {f"{n}_self": float(act[i].mean())
                          for i, n in enumerate(self.names)})
