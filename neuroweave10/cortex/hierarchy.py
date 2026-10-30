"""Кортикальная иерархия (§II.2.3): зоны на градиенте τ(g).

SENS(g=0, τ≈20мс) → ASSOC(g=0.5, τ≈100мс) → PFC(g=0.9, τ≈1-2с).
FF: выход зоны k → basal зоны k+1 (через HO-трансталамический путь).
FB: выход зоны k+1 → apical зоны k (NMDA, предсказание сверху).
GW-бродкаст идёт в апикал всех зон.

Внутренняя метрика: error = S - φ(V*_D) по каждой зоне — несоответствие
между спайком и дендритным «обещанием» (предиктивное кодирование
Rao & Ballard 1999 внутри нейрона).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from ..config import GenesisConfig
from ..core.sdr import LCASparseEncoder
from .column import CorticalColumn, ColumnOut


@dataclass
class HierarchyOut:
    zones: Dict[str, ColumnOut]
    errors: Dict[str, torch.Tensor]
    l5: Dict[str, torch.Tensor]     # L5-драйверы каждой зоны
    spikes_all: torch.Tensor        # конкатенация для монитора критичности


class CorticalHierarchy:
    """Цепочка зон SENS→ASSOC→PFC с FF/FB и градиентом τ(g)."""

    def __init__(self, cfg: GenesisConfig, batch: int = 1):
        self.cfg = cfg
        self.batch = batch
        zs = cfg.zones
        self.encoder = LCASparseEncoder(cfg.input_dim, cfg.encoder, cfg.seed)
        self.cols: List[CorticalColumn] = []
        for i, z in enumerate(zs):
            n_in = cfg.encoder.n_sdr if i == 0 else zs[i - 1].n_neurons
            n_fb = zs[i + 1].n_neurons if i < len(zs) - 1 else cfg.gw.d_gw
            self.cols.append(CorticalColumn(z, n_in, n_fb, cfg.neuron,
                                            cfg.synapse, seed=cfg.seed + i))
        self._prev: Dict[str, torch.Tensor] = {}

    def reset_state(self, batch: int, device=None, dtype=torch.float32) -> None:
        self.batch = batch
        for c in self.cols:
            c.reset_state(batch, device, dtype)
        self._prev = {}

    # ------------------------------------------------------------------
    def step(self, x: torch.Tensor, gw_broadcast: Optional[torch.Tensor],
             ho_drive: Dict[str, torch.Tensor],
             f_mod: float, vip_per_zone: Dict[str, float],
             learn: bool = True, dt: float = 1.0) -> HierarchyOut:
        """Параллельный такт: каждая зона получает FF (prev L5/SDR через HO),
        FB (следующая зона, NMDA) и GW-бродкаст в апикал."""
        zs, outs, errs, l5s = self.cfg.zones, {}, {}, {}
        spikes = []
        for i, col in enumerate(self.cols):
            z = zs[i]
            # FF: SDR для SENS; трансталамический HO-драйв для остальных
            if i == 0:
                x_ff = self.encoder.encode(x, learn=learn)
            else:
                x_ff = ho_drive.get(z.name, torch.zeros(
                    self.batch, zs[i - 1].n_neurons))
            # FB: апикальный вход от следующей зоны или GW
            if i < len(self.cols) - 1:
                nxt = self._prev.get(zs[i + 1].name)
                x_fb = nxt if nxt is not None else torch.zeros(
                    self.batch, zs[i + 1].n_neurons)
            else:
                x_fb = gw_broadcast if gw_broadcast is not None else \
                    torch.zeros(self.cfg.gw.d_gw)
            if x_fb.dim() == 1:
                x_fb = x_fb.unsqueeze(0).expand(self.batch, -1)
            if x_ff.dim() == 1:
                x_ff = x_ff.unsqueeze(0).expand(self.batch, -1)
            out = col.step(x_ff, x_fb, f_mod, vip_per_zone.get(z.name, 0.0),
                           learn=learn, dt=dt)
            self._prev[z.name] = out.s
            outs[z.name], errs[z.name] = out, out.error
            l5s[z.name] = out.l5_drive
            spikes.append(out.s)
        return HierarchyOut(zones=outs, errors=errs, l5=l5s,
                            spikes_all=torch.cat(spikes, dim=1))

    def consolidate(self, factor: float = 0.5) -> None:
        for c in self.cols:
            c.consolidate(factor)
