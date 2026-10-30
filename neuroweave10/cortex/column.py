"""Кортикальная колонка V10 (§II.2.1-2.2).

Микроцепь:
  PYR (US-нейроны): базальный дендрит ← AMPA-FF от предыдущей зоны;
                    апикальный ← NMDA-FB от следующей зоны + GW;
  PV — перисоматическое торможение, k-WTA на сому (быстрая реакция);
  SST — дендритное торможение на апикальный вход (гейтинг FB);
  VIP — тормозит SST → дизингибиция апикала. g = g_0/(1+([VIP]/K_d)^n).

Обучение: трёхфакторное правило с внутренней ошибкой зоны
error_i = S_i - φ(V*_D,i) (Rao & Ballard 1999 внутри дендрита).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from ..config import USNeuronCfg, USSynapseCfg, ZoneCfg
from ..core.neuron import USNeuronPool, NeuronOut
from ..core.synapse import GenesisSynapse


@dataclass
class ColumnOut:
    s: torch.Tensor           # спайки пирамидальных
    error: torch.Tensor       # внутренняя ошибка зоны
    l5_drive: torch.Tensor    # выход L5-подмножества → HO/BG
    v_s: torch.Tensor
    ca: torch.Tensor          # Ca2+ → 2-AG/NO
    rate_d: torch.Tensor


class CorticalColumn:
    """Зона на градиенте: PYR + PV + SST + VIP + FF/FB/REC-синапсы."""

    def __init__(self, zcfg: ZoneCfg, n_in: int, n_fb: int,
                 ncfg: USNeuronCfg, scfg: USSynapseCfg, seed: int = 0):
        self.zcfg = zcfg
        self.name = zcfg.name
        n = zcfg.n_neurons
        self.pyr = USNeuronPool(n, ncfg, name=zcfg.name)
        self.pv = torch.zeros(zcfg.n_pv)
        self.sst = torch.zeros(zcfg.n_sst)
        self.vip = torch.zeros(zcfg.n_vip)
        self.vip_drive = 0.0                      # [VIP] для дизингибиции
        # Синапсы: FF (пред→basal), FB (след→apical), рекуррентные, к интернейронам
        self.w_ff = GenesisSynapse(n_in, n, scfg, f"{zcfg.name}_ff", seed,
                                   init_scale=1.2)
        self.w_fb = GenesisSynapse(n_fb, n, scfg, f"{zcfg.name}_fb", seed + 1)
        self.w_rec = GenesisSynapse(n, n, scfg, f"{zcfg.name}_rec", seed + 2)
        self.w_pv = torch.randn(n, zcfg.n_pv) * 0.3        # PYR→PV
        self.w_pv_in = torch.randn(zcfg.n_pv, n).abs() * 0.4  # PV→PYR (soma)
        self.w_sst_in = torch.randn(zcfg.n_sst, n).abs() * 0.3  # SST→PYR (apical)
        self.w_vip_in = torch.randn(zcfg.n_vip, zcfg.n_sst).abs() * 0.5
        self.kd_vip, self.n_vip_hill = 0.4, 2.0
        # L5-подмножество (нижняя треть пула) → HO-драйверы
        self.l5_idx = torch.arange(2 * n // 3, n)
        # Базальное дендритное дерево Poirazi: суперлинейные ветви+плато
        from ..core.dendrite import DendriticTree
        self.dendrite = DendriticTree(n, n_in, n_branch=4, seed=seed + 3)

    def reset_state(self, batch: int, device=None, dtype=torch.float32) -> None:
        self.pyr.reset_state(batch, device, dtype)
        self.dendrite.reset_state(batch, device, dtype)

    # ------------------------------------------------------------------
    def step(self, x_ff: torch.Tensor, x_fb: torch.Tensor,
             f_mod: float, vip_mod: float,
             ltp_field: Optional[torch.Tensor] = None,
             learn: bool = True, dt: float = 1.0) -> ColumnOut:
        """x_ff [B,n_in] → basal AMPA; x_fb [B,n_fb] → apical NMDA."""
        g_ampa = self.w_ff.drive(x_ff).clamp(0.0, None) * 4.0
        g_nmda = self.w_fb.drive(x_fb).clamp(0.0, None)
        g_rec = self.w_rec.drive(self._last_s) \
            if hasattr(self, "_last_s") else torch.zeros_like(g_ampa)

        # --- Интернейроны ---
        pyr_s = self._last_s if hasattr(self, "_last_s") \
            else torch.zeros(x_ff.shape[0], self.zcfg.n_neurons)
        pv_act = torch.relu(pyr_s @ self.w_pv)
        self.pv = pv_act.mean(0)
        # k-WTA: PV оставляет топ-k пирамидальных, давит остальные
        k = max(1, int(0.3 * self.zcfg.n_neurons))
        thr = pyr_s.topk(k, dim=1).values[:, -1:]
        g_pv = (pv_act @ self.w_pv_in) * (pyr_s < thr).float()
        # SST: тормозит апикал; VIP дизингибирует: g=g0/(1+([VIP]/Kd)^n)
        vip = vip_mod
        sst_gate = 1.0 / (1.0 + (vip / self.kd_vip) ** self.n_vip_hill)
        sst_act = torch.relu(g_nmda @ self.w_sst_in.t() * 0.1)
        g_sst = (sst_act @ self.w_sst_in) * sst_gate
        g_nmda = g_nmda * sst_gate              # SST гейтит апикал
        g_gaba = (g_pv + g_sst).clamp(0.0, 5.0)

        # Базальный вход идёт через ветви: суперлинейная сумма + плато
        pyr_s = self._last_s if hasattr(self, "_last_s") \
            else torch.zeros_like(g_ampa)
        g_tree = self.dendrite.step(x_ff, pyr_s)
        g_ampa = (g_ampa + g_tree).clamp(0.0, None)
        # Тонический фон: кортикальные нейроны получают постоянный
        # неспецифический вход, удерживающий сому вблизи порога.
        i_ext = torch.full_like(g_ampa, 1.5)
        mod_ctx = self._mod_ctx if isinstance(
            getattr(self, "_mod_ctx", None), dict) else {}
        out = self.pyr.step(g_ampa + g_rec * 0.5, g_nmda, g_gaba, i_ext,
                            ach=mod_ctx.get("ach", 0.0),
                            histamine=mod_ctx.get("histamine", 0.0),
                            nmda_gain=mod_ctx.get("nmda_gain", 1.0),
                            p_rel=mod_ctx.get("p_rel", 1.0),
                            vth_shift=mod_ctx.get("vth_shift", 0.0),
                            gaba_tonic=mod_ctx.get("gaba_tonic", 1.0))
        self._last_s = out.s

        if learn:
            self.w_ff.learn(out.error, x_ff, out.s, f_mod, ltp_field, dt)
            self.w_fb.learn(out.error, x_fb, out.s, f_mod, ltp_field, dt)
            self.w_rec.learn(out.error, self._last_s, out.s,
                             f_mod, ltp_field, dt)
        return ColumnOut(s=out.s, error=out.error,
                         l5_drive=out.s[:, self.l5_idx],
                         v_s=out.v_s, ca=out.ca, rate_d=out.rate_d)

    def consolidate(self, factor: float = 0.5) -> None:
        """Сон: перенос быстрых изменений в медленные веса."""
        for w in (self.w_ff, self.w_fb, self.w_rec):
            w.w *= 1.0 - factor * 0.0  # w уже единый; усиливаем маску
            w.u_syn.fill_(1.0)
