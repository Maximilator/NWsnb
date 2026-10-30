"""Таламическая ретикулярная ядро TRN (§II.1.3).

Секторный GABA-шлюз: каждый сектор TRN контролирует свой релейный канал.
Управление:
  L6-кортикоталамический — выбор «что пропустить»;
  PFC/AST — нисходящий приоритет внимания (searchlight, Crick 1984);
  латеральное торможение WTM между секторами.

g_k = g_L6 · g_AST · (1 - lateral_inh); итоговый гейт FO: Π g_k.
"""
from __future__ import annotations

from typing import Optional

import torch


class TRN:
    """Ретикулярное ядро: посекторный GABA-гейт + латеральное торможение."""

    def __init__(self, n_sectors: int):
        self.n = n_sectors
        self.act = torch.zeros(n_sectors)      # активность секторов
        self.w_lat = torch.ones(n_sectors, n_sectors) * 0.3
        self.w_lat.fill_diagonal_(0.0)         # латеральное WTM
        self.gate = torch.ones(n_sectors)      # выходной гейт на релей

    def step(self, l6_drive: torch.Tensor,
             ast_priority: Optional[torch.Tensor] = None) -> torch.Tensor:
        """l6_drive [S] — кортикоталамический запрос каждого сектора.

        Возвращает gate [S]: 0=закрыт, 1=открыт."""
        x = l6_drive.clamp(0.0, 1.0)
        if ast_priority is not None:
            x = x * (0.5 + ast_priority.clamp(0.0, 1.0))
        self.act = self.act * 0.9 + x * 0.1
        lateral = self.act @ self.w_lat        # торможение соседей
        self.gate = (self.act - lateral).clamp(0.0, 1.0)
        return self.gate
