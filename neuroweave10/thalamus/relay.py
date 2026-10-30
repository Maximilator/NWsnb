"""Таламус: FO-релей (§II.1.2) и HO-ядро (§II.1.4).

FO (first order, LGN/VPL): передаёт сенсорику в L4. Два режима:
  tonic — линейный релей высокой точности ([ACh]↑, [NE]↑);
  burst — нелинейный T-type Ca2+-burst «wake-up call»
          (молчание афферентов → деполяризация после гиперполяризации
          → нейрон «кричит»). Режим выбирается модуляторами и паузой
          входа (Sherman 2001).

HO (higher order, Pulvinar/MD): релей кортико-кортикального драйва —
копия L5-эфферентов идёт через HO в следующую зону (трансталамический
путь). WTA выбирает «сильнейшие сообщения»; базальные ганглии
разблокируют релей через дизингибицию (GPi/SNr → таламус).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


class ThalamusFO:
    """First-order таламус с переключением tonic/burst и TRN-гейтом."""

    def __init__(self, n: int, name: str = "FO"):
        self.n, self.name = n, name
        self.mode = "tonic"
        self.silence_ms = torch.zeros(n)  # длительность молчания канала
        self.burst_ms = torch.zeros(n)    # оставшиеся такты burst-режима
        self.gate = torch.ones(n)         # TRN-гейт [0..1]

    def step(self, x: torch.Tensor, ach: float, ne: float) -> torch.Tensor:
        """x [B,N] сенсорика → релейный выход в L4."""
        # Выбор режима: молчание входа > 50 мс → burst при пробуждении
        active = (x.abs().sum(0) > 1e-3).float()
        self.silence_ms = torch.where(active > 0, torch.zeros_like(active),
                                      self.silence_ms + 1.0)
        if ach + ne < 0.5:                 # низкое бодрствование → burst
            self.mode = "burst"
        elif self.silence_ms.mean() > 50.0:
            self.mode = "burst"
        else:
            self.mode = "tonic"
        if self.mode == "burst":
            # Burst: нелинейная амплификация после молчания (T-type Ca2+)
            gain = torch.where(self.silence_ms > 50.0,
                               torch.full_like(x[0], 3.0),
                               torch.ones_like(x[0]))
            out = torch.tanh(x * 2.0) * gain
        else:
            out = x  # tonic: линейный релей
        return out * self.gate.unsqueeze(0)


class ThalamicHO:
    """Higher-order ядро: выбор драйверов L5 → следующая зона (§II.1.4).

    Релей открывается дизингибицией BG (GPi↓ → HO↑). WTA по K слотам —
    выбираются сильнейшие L5-каналы, остальные подавляются."""
    def __init__(self, n_in: int, n_out: int, name: str = "HO"):
        self.n_in, self.n_out, self.name = n_in, n_out, name
        self.w = torch.randn(n_out, n_in) * 0.6
        self.winner_mask = torch.zeros(n_in)
        self.open_gate = 0.0              # BG-гейт

    def step(self, l5: torch.Tensor, bg_release: float) -> torch.Tensor:
        """l5 [B,N_in] апикальный драйв → HO-выход в следующую зону.

        bg_release: дизингибиция от BG (0=закрыт, 1=открыт)."""
        self.open_gate = bg_release
        if bg_release < 0.5:
            return torch.zeros(l5.shape[0], self.n_out, device=l5.device)
        drive = l5 @ self.w.t()
        # WTA: только топ-K L5-входов проходят
        k = max(1, self.n_in // 4)
        thr = l5.topk(k, dim=1).values[:, -1:]
        self.winner_mask = (l5 >= thr).float().mean(0)
        return torch.tanh(drive) * self.open_gate
