"""Глобальное рабочее пространство (§II.4.1): Modern Hopfield Network.

Слоты K=4 (Cowan 2001) — когерентные пакеты информации d_GW.
Ignition: нелинейная фазовая транзиция (Dehaene & Changeux 2011):
  A = σ(γ_ign·(S - S_th)),  S = метрика внимания кандидата
  [NE] сдвигает порог: более резкий переход при NE↑.

Извлечение (Ramsauer et al. 2021):
  ξ_new = X·softmax(β_GW·Xᵀξ)    — один шаг Hopfield-обновления
  E(ξ)  = -½ξᵀXᵀXξ               — Lyapunov-энергия (монотонно ↓)

Бродкаст: чтение K-слотов всеми зонами (дешёвое чтение).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from ..config import WorkspaceCfg


@dataclass
class GWOut:
    broadcast: torch.Tensor      # [d_gw] общий сигнал зонам
    content: List[torch.Tensor]  # активные слоты
    ignition: float              # A ∈ [0,1]
    winner_zone: int             # какая зона победила
    energy: float                # Lyapunov-энергия


class GlobalWorkspace:
    """GNW как современная сеть Хопфилда с ignition-гейтом."""

    def __init__(self, cfg: WorkspaceCfg):
        self.cfg = cfg
        self.x = torch.zeros(cfg.n_slots, cfg.d_gw)   # слоты-аттракторы
        self.gate = torch.zeros(cfg.n_slots)          # доступ слота
        self.ignition = 0.0

    # ------------------------------------------------------------------
    def compete(self, candidates: List[torch.Tensor],
                ne: float) -> GWOut:
        """candidates — пакеты от зон (по одному d_gw-вектору на зону)."""
        c = self.cfg
        if not candidates:
            self.x *= c.decay
            return GWOut(torch.zeros(c.d_gw), [], 0.0, -1, 0.0)
        stack = torch.stack([c_.detach().float().flatten()[:c.d_gw]
                             for c_ in candidates])
        stack = torch.nn.functional.pad(
            stack, (0, max(0, c.d_gw - stack.shape[1])))[:, :c.d_gw]
        # S = внимание-кандидата = норма пакета; ignition по sigmoid
        scores = stack.norm(dim=1)
        beta = c.beta_gw * (1.0 + 0.5 * ne)   # NE ↑ крутизну
        a = torch.sigmoid(c.gamma_ign * (scores - c.s_th))
        self.ignition = float(a.max())
        winner = int(a.argmax()) if self.ignition > 0.5 else -1
        if winner >= 0:
            # Победивший пакет занимает свободный/слабейший слот
            slot = int(self.gate.argmin()) if (self.gate > 0).any() else 0
            self.x[slot] = stack[winner]
            self.gate[slot] = self.ignition
        self.gate *= c.decay
        # Извлечение: ξ_new = X·softmax(β Xᵀ ξ)
        energy = 0.0
        for s in range(c.n_slots):
            if self.gate[s] <= 1e-3:
                continue
            xi = self.x[s]
            for _ in range(c.retrieval_steps):
                logits = beta * (self.x @ xi)
                xi_new = self.x.t() @ torch.softmax(logits, dim=0)
                xtx_xi = self.x.t() @ (self.x @ xi_new)
                energy_new = -0.5 * float((xi_new * xtx_xi).sum())
                xi = xi_new
            self.x[s] = xi
            energy = energy_new
        broadcast = (self.x * self.gate.unsqueeze(1)).sum(0)
        return GWOut(broadcast=broadcast,
                     content=[self.x[i] for i in range(c.n_slots)
                              if self.gate[i] > 1e-3],
                     ignition=self.ignition, winner_zone=winner,
                     energy=energy)
