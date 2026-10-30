"""Мультикомпартментный пирамидальный нейрон Urbanczik-Senn (§1.2).

Биологическое ядро V10: сома + базальный дендрит (AMPA, feedforward)
+ апикальный дендрит (NMDA, feedback) — три связанных компартмента:

  C_m dV_S/dt  = -g_L(V_S-E_L) + g_D(V_D-V_S) + g_A(V_A-V_S)
                 - I_fast - I_metabolic
  C_m dV_D/dt  = -g_L(V_D-E_L) + g_D(V_S-V_D) + g_D AMPA(E_exc-V_D)
  C_m dV_A/dt  = -g_L(V_A-E_L) + g_A(V_S-V_A) + g_A NMDA·B(V_A)·(E_exc-V_A)

  I_fast       = g_GABA·(1+η_ALLO[ALLO])·(V_S-E_Cl)   — быстрое торможение
  I_metabolic  = k_ATP·(1-E_i)·(V_S-E_K) + k_GIRK·[Ado]_i·(V_S-E_K)
  B(V)         = 1/(1 + [Mg2+]/3.57·exp(-0.062·V))    — Jahr-Stevens 1990
  s_i(t)       = H(V_S - Θ);  Θ адаптируется как ALIF (Bellec 2018)
  φ(V)         = φ_max/(1+exp(-β(V-Θ)))               — S-функция частоты
  Ca2+         = экспоненциальный след спайков → 2-AG, NO, BDNF-пути

Два сигнала на нейрон: «обещание» дендрита φ(V*_D) для обучения и
истинный спайк S_i. Ошибка S_i-φ(V*_D,i) — внутренняя метрика зоны
(§1.3: обучение без обратного распространения).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from ..config import USNeuronCfg


def mg_block(v: torch.Tensor, mg: float = 1.0) -> torch.Tensor:
    """Jahr-Stevens: B(V) = 1/(1 + [Mg2+]/3.57·exp(-0.062V))."""
    return 1.0 / (1.0 + (mg / 3.57) * torch.exp(-0.062 * v))


@dataclass
class NeuronOut:
    s: torch.Tensor          # [B,N] спайки {0,1}
    v_s: torch.Tensor
    v_d: torch.Tensor        # базальный дендрит
    v_a: torch.Tensor        # апикальный дендрит
    rate_d: torch.Tensor     # φ(V*_D) — предсказание дендрита
    error: torch.Tensor      # S_i - φ(V*_D) — сигнал внутренней ошибки
    energy: torch.Tensor     # [B,N] ATP
    ca: torch.Tensor         # [B,N] постсинаптический Ca2+


class USNeuronPool:
    """Пул мультикомпартментных нейронов. Состояние — плоские тензоры."""

    def __init__(self, n: int, cfg: USNeuronCfg, name: str = "pyr"):
        self.n, self.cfg, self.name = n, cfg, name
        self.register_buffer = None  # не nn.Module — лёгкие пулы

    def reset_state(self, batch: int, device=None, dtype=torch.float32) -> None:
        c = self.cfg
        f = lambda *s: torch.zeros(batch, *s, device=device, dtype=dtype)
        self.v_s = f(self.n).fill_(c.e_l)
        self.v_d = f(self.n).fill_(c.e_l)
        self.v_a = f(self.n).fill_(c.e_l)
        self.theta = f(self.n).fill_(c.theta)   # адаптивный порог Θ_i
        self.refr = f(self.n)                   # такты рефрактерности
        self.ca = f(self.n)                     # Ca2+-след
        self.energy = f(self.n).fill_(1.0)      # ATP E_i ∈ [0,1]
        self.ado_i = f(self.n)                  # локальный аденозин A1
        self.allo = f(self.n)                   # нейростероид ALLO
        self.noise = f(self.n)
        self._spikes_hist: list = []

    # ------------------------------------------------------------------
    def phi(self, v: torch.Tensor, theta: Optional[torch.Tensor] = None) -> torch.Tensor:
        """φ(V) = φ_max·σ(β(V-Θ)) — частота, Гц."""
        th = self.theta if theta is None else theta
        return self.cfg.phi_max * torch.sigmoid(self.cfg.beta * (v - th))

    def v_star_d(self) -> torch.Tensor:
        """V*_D = (g_L·E_L + g_D·V_D)/(g_L+g_D) — потенциал дендрита без спайка.

        Это «обещание» апикального дендрита; используется трёхфакторным
        правилом как предсказание сомы (Urbanczik & Senn 2014)."""
        c = self.cfg
        return (c.g_l * c.e_l + c.g_d * self.v_d) / (c.g_l + c.g_d)

    def step(self,
             g_ampa: torch.Tensor,      # [B,N] открытая проводимость AMPA
             g_nmda: torch.Tensor,      # [B,N] проводимость NMDA (апикал)
             g_gaba: torch.Tensor,      # [B,N] проводимость GABA
             i_ext: Optional[torch.Tensor] = None,
             ach: float = 0.0,          # ACh: M-ток K_v7↓ → Θ↓ (encode-режим)
             histamine: float = 0.0,    # H1: sAHP-блок → берстовая активность
             nmda_gain: float = 1.0,    # Φ_DSer·Ψ_Zn·DHEA ко-агонистный гейт
             p_rel: float = 1.0,        # P_release-масштаб (NPY/Dyn/AEA/Gal)
             vth_shift: float = 0.0,    # E2-мембранный сдвиг Θ (мВ)
             gaba_tonic: float = 1.0,   # ALLO: GABA_A PAM
             ) -> NeuronOut:
        """Один шаг интеграции (dt=1 мс), все токи сразу."""
        c = self.cfg
        dt = c.dt
        B = self.v_s

        # --- дендриты ---
        # Базальный: AMPA-вход (возбуждение) + связь с сомой
        i_d = c.g_d * (B - self.v_d) + g_ampa * p_rel * (c.e_exc - self.v_d)
        dv_d = (-c.g_l * (self.v_d - c.e_l) + i_d) / c.c_m
        self.v_d = self.v_d + dt * (dv_d / c.tau_d)

        # Апикальный: NMDA·(D-ser·Zn·DHEA-гейт)·B(V_A)·(E_exc−V_A)
        b_v = mg_block(self.v_a, c.mg)
        i_a = c.g_a * (B - self.v_a) \
            + g_nmda * nmda_gain * b_v * (c.e_exc - self.v_a)
        dv_a = (-c.g_l * (self.v_a - c.e_l) + i_a) / c.c_m
        self.v_a = self.v_a + dt * (dv_a / c.tau_d_apical)

        # --- сома: проводимости (не токи — Закон 3, шунт-физика) ---
        # GABA: шунтирующая проводимость к E_Cl; ALLO усиливает её
        # (тонический PAM + нейростероидный буфер — сон/анксиолиз).
        g_inh = g_gaba * gaba_tonic * (1.0 + c.allo_gain * self.allo)
        # Метаболический тормоз: KATP (ATP↓) + GIRK (Ado A1) → V_S к E_K.
        g_met = c.k_atp * (1.0 - self.energy) + c.k_girk * self.ado_i
        # GPCR-K+-токи (§3.1): M-ток подавляется ACh (mAChR→I_M↓→Θ↓),
        # sAHP растёт по Ca2+ и блокируется гистамином H1.
        g_m = 0.05 / (1.0 + ach / 0.5)          # M-current (K_v7)
        g_ahp = 0.08 * self.ca * (1.0 - histamine / (histamine + 0.4))
        g_gpcr = g_m + g_ahp
        ext = 0.0 if i_ext is None else i_ext
        if c.stochastic:
            # OU-шум Орнштейна-Уленбека: коррелированный во времени
            # мембранный шум (τ_ou≈10 мс) — биологическая стохастичность
            # фазы обучения, заменяет алгоритмический Scheduled Sampling.
            self.noise = self.noise * (1.0 - dt / c.tau_ou) + \
                torch.randn_like(B) * (c.sigma * (dt / c.tau_ou) ** 0.5)
        else:
            self.noise = torch.zeros_like(B)

        # --- сома: EXACT-STEP по проводимостям (REGENESIS, Закон 3) ---
        # GABA — шунтирующая проводимость к E_Cl≈-75 мВ, не ток-вычитание.
        # Точное решение C_m dV/dt = -Σ_i g_i(V-E_i) + I_ext за шаг dt:
        #   V_inf = (Σ_i g_i·E_i + I_ext)/Σ_i g_i   — целевой потенциал
        #   τ_eff = C_m/Σ_i g_i                     — эффективная константа
        #   V ← V_inf + (V−V_inf)·exp(−dt/τ_eff)
        # Торможение шунтирует: при g_inh→∞ V→E_Cl (не ниже); возбуждение
        # прижато к E_exc. Никакой гиперполяризации ниже реверсии хлора.
        g_inh_tot = g_inh + g_met + g_gpcr  # все тормозные проводимости
        g_exc_tot = c.g_d + c.g_a           # связи с дендритами (к их V)
        g_tot = c.g_l + g_exc_tot + g_inh_tot
        v_inf = (c.g_l * c.e_l + c.g_d * self.v_d + c.g_a * self.v_a
                 + g_inh * c.e_cl + (g_met + g_gpcr) * c.e_k
                 + ext + self.noise) / g_tot.clamp(min=1e-6)
        tau_eff = c.c_m / g_tot.clamp(min=1e-6)
        self.v_s = v_inf + (B - v_inf) * torch.exp(-dt / tau_eff)

        # --- спайк: порог Θ_i (адаптивный) + рефрактерность ---
        # E2-мембранный сдвиг порога и M-ток (ACh) понижают Θ;
        # аденозин A1 (продукт распада ATP) ПОВЫШАЕТ Θ — честная
        # усталость организма (Закон 5: энергия регулирует, не логирует).
        theta_eff = self.theta - vth_shift - 1.5 * ach \
            + c.ado_theta_gain * self.ado_i
        s = (self.v_s > theta_eff).float() * (self.refr <= 0).float()
        self.v_s = self.v_s - s * (self.theta - c.e_l) * 0.5
        self.refr = (self.refr - 1.0).clamp(min=0.0) + s * c.refractory

        # ALIF-адаптация порога: Θ ← Θ + β·s - (Θ-Θ0)/τ_adapt
        self.theta = self.theta + c.beta_adapt * s \
            - dt * (self.theta - c.theta) / c.tau_adapt

        # Ca2+-след спайка (постсинаптический — драйвер 2-AG/NO/BDNF)
        self.ca = self.ca * (1.0 - dt / c.tau_ca) + s

        # ATP-метаболизм: спайк тратит энергию; KATP/GIRK тормозят при дефиците
        self.energy = (self.energy - s * 0.004).clamp(min=0.0, max=1.0)

        rate_d = self.phi(self.v_star_d())
        return NeuronOut(s=s, v_s=self.v_s, v_d=self.v_d, v_a=self.v_a,
                         rate_d=rate_d, error=s - rate_d / c.phi_max,
                         energy=self.energy, ca=self.ca)
