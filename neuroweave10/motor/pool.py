"""Спайковый моторный пул (REGENESIS, Законы 1–2).

Заменяет MLP-декодер: `vocab` нейронов с проводимостной мембраной
(exact-step, Закон 3), ALIF-адаптацией порога и OU-шумом. Входы:

  * w_l5   — афференты от слоя L5 коры (AMPA-проводимость + STP),
  * w_hip  — гиппокамп CA1 → апикальный NMDA-вход с Mg²⁺-блоком
             (Jahr-Stevens): без базального драйва гиппокамп лишь
             «праймит» мембрану, спайк не вызывает,
  * w_lap  — медленные шкалы Лапласова банка → сдвиг возбудимости.

Выбор токена — Winner-Take-All: нейрон, выстреливший первым / чаще за
суб-такты такта, и есть выходное слово. Никакого argmax по логитам
обучаемого MLP — решение принимает спайковая гонка.

Закон 2: кратковременная депрессия Tsodyks-Markram на афферентах —
при повторном выстреле нейрона его пресинаптические везикулы истощаются
(x ← x(1−u·x), восстановление τ_D≈200 мс) и эффективность выхода падает
(out ~ 1→0.2 за ~25 мс). Петля «found found found» гасится биофизикой,
а не алгоритмическим вето.

Обучение — трёхфакторное: ошибка нейрона err_i = (i==target)·1 − act_i
(nudging в духе Urbanczik-Senn) × пре-след L5 × f_mod; пластичность
замораживается при аденилатном долге выше порога (Закон 5).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from ..config import MotorCfg
from ..core.neuron import mg_block


@dataclass
class MotorOut:
    token: int                 # выбранный токен (WTA-победитель)
    spikes: torch.Tensor       # [B,V] спайки моторных нейронов
    v: torch.Tensor            # [B,V] мембранный потенциал
    margin: float              # разрыв V между 1-м и 2-м кандидатом, мВ
    stp_depleted: float        # средний ресурс x на афферентах победителя
    firing_rate: torch.Tensor  # [V] счёт спайков за такт


class MotorPool:
    """Пул спайковых моторных нейронов размером со словарь."""

    def __init__(self, vocab_size: int, n_l5: int, n_hipp: int,
                 n_lap: int, cfg: MotorCfg, seed: int = 0):
        self.cfg = cfg
        self.vocab, self.n_l5 = vocab_size, n_l5
        g = torch.Generator().manual_seed(seed)
        self.w_l5 = torch.randn(vocab_size, n_l5, generator=g) * 0.35
        self.w_hip = torch.randn(vocab_size, n_hipp, generator=g) * 0.20
        self.w_lap = torch.randn(vocab_size, n_lap, generator=g) * 0.10
        # Рекуррентные связи моторной коры (как в реальном M1)
        self.w_rec_m = torch.randn(vocab_size, vocab_size,
                                   generator=g) * 0.05
        self.w_rec_m.fill_diagonal_(0.0)
        # Базальные ганглии → моторный гейт (n_actions каналов выбора)
        self.n_bg = 0
        self.w_bg = None
        # STP Tsodyks-Markram: ресурс x и утилизация u на каждый афферент
        self.stp_x = torch.ones(vocab_size, n_l5)
        self.stp_u = torch.ones(vocab_size, n_l5) * cfg.stp_u0
        # Усталость выхода моторного нейрона (аксональная депрессия)
        self.out_fatigue = torch.ones(vocab_size)
        # Адаптивный STP: частота афферента → глубина депрессии и τ_rec
        self.rate_aff = torch.zeros(n_l5)
        self.rate_post = torch.zeros(vocab_size)
        # Следы трёхфакторного обучения
        self.z_pre = torch.zeros(n_l5)
        self.z_post = torch.zeros(vocab_size)
        self.eligibility = torch.zeros(vocab_size, n_l5)
        self._last_counts = torch.zeros(vocab_size)
        self._bg_decision: Optional[torch.Tensor] = None

    def attach_basal_ganglia(self, n_actions: int, seed: int = 0) -> None:
        """BG→моторная кора: решение DDM открывает канал действия."""
        g = torch.Generator().manual_seed(seed)
        self.n_bg = n_actions
        self.w_bg = torch.randn(self.vocab, n_actions, generator=g) * 0.15

    def reset_state(self, batch: int, device=None, dtype=torch.float32):
        f = lambda *s: torch.zeros(batch, *s, device=device, dtype=dtype)
        c = self.cfg
        self.v = f(self.vocab).fill_(c.e_l)
        self.theta = f(self.vocab).fill_(c.theta0)
        self.refr = f(self.vocab)
        self.ou = f(self.vocab)
        self.counts = f(self.vocab)

    # ------------------------------------------------------------------
    def _sub_step(self, s_l5: torch.Tensor, hipp_v: Optional[torch.Tensor],
                  lap_slow: Optional[torch.Tensor], dt: float,
                  ou_on: bool) -> None:
        """Один мс интеграции проводимостной мембраны + STP-обновление."""
        c = self.cfg
        # Афферентный драйв L5: проводимость × STP-ресурс × усталость выхода
        w_eff = self.w_l5 * self.stp_x * self.stp_u
        g_l5 = (s_l5 @ w_eff.t()).clamp(min=0.0) * 2.0
        g_l5 = g_l5 * self.out_fatigue.unsqueeze(0)
        # Рекуррентный контур моторной коры: спайки соседей модулируют
        # мембрану — самоорганизация последовательностей
        if self._last_counts is not None:
            g_l5 = g_l5 + (self._last_counts.unsqueeze(0)
                           @ self.w_rec_m.t()).clamp(min=0.0) * 0.3
        # Базальные ганглии: решение DDM открывает канал действия
        if self.w_bg is not None and self._bg_decision is not None:
            g_l5 = g_l5 + (self._bg_decision.unsqueeze(0)
                           @ self.w_bg.t()).clamp(min=0.0)
        # Гиппокамп: только через Mg²⁺-блок NMDA — прайминг, не спайк
        g_hipp = torch.zeros_like(g_l5)
        if hipp_v is not None:
            b_v = mg_block(self.v)                       # Jahr-Stevens
            g_hipp = (hipp_v @ self.w_hip.t()).clamp(min=0.0) * b_v
        # Лапласовы медленные шкалы: сдвиг возбудимости (не драйв)
        th_shift = torch.zeros_like(g_l5)
        if lap_slow is not None:
            th_shift = (lap_slow @ self.w_lap.t()).clamp(min=-4.0, max=4.0)
        # OU-шум фазы обучения вместо Scheduled Sampling (Закон, шаг 5)
        if ou_on:
            self.ou = self.ou * (1.0 - dt / 10.0) + torch.randn_like(
                self.v) * (c.sigma_ou * (dt / 10.0) ** 0.5)
        # Глобальное делительное торможение: WTA-пул PV —
        # торможение ∝ суммарному возбуждению; сильнейший пробивается.
        g_wta = c.wta_inh * g_l5.mean(dim=1, keepdim=True)
        g_inh = g_wta.expand_as(g_l5) * 0.2
        # Exact-step (Закон 3): V_inf = (Σ g·E + I)/Σg
        g_tot = c.g_l + g_l5 + g_hipp + g_inh
        v_inf = (c.g_l * c.e_l + (g_l5 + g_hipp) * c.e_exc
                 + g_inh * c.e_cl + self.ou) / g_tot.clamp(min=1e-6)
        tau_eff = c.tau_m / g_tot.clamp(min=1e-6)
        self.v = v_inf + (self.v - v_inf) * torch.exp(-dt / tau_eff)
        # Спайк: порог + рефрактерность; ALIF-адаптация Θ
        theta_eff = self.theta - th_shift
        s = (self.v > theta_eff).float() * (self.refr <= 0).float()
        self.v = self.v - s * (self.theta - c.e_l) * 0.5
        self.refr = (self.refr - 1.0).clamp(min=0.0) + s * c.refractory
        self.theta = self.theta + c.beta_adapt * s \
            - dt * (self.theta - c.theta0) / c.tau_adapt
        self.counts += s
        # Адаптивный STP Tsodyks-Markram: история частоты афферента
        # задаёт глубину депрессии и время восстановления — частый вход
        # истощается глубже (u↑→быстрый расход x) и восстанавливается
        # через ~25 мс, не мгновенно.
        self.rate_aff = self.rate_aff * 0.98 + s_l5.mean(0) * 0.02
        self.rate_post = self.rate_post * 0.98 + s.mean(0) * 0.02
        u_eff = (c.stp_u0 * (1.0 + 2.0 * self.rate_aff)).clamp(max=0.95)
        tau_d_eff = c.stp_tau_d / (1.0 + 3.0 * self.rate_aff)  # → ~25-30 мс
        act = s_l5.mean(0).clamp(0.0, 1.0)                   # [n_l5]
        u_rel = u_eff.unsqueeze(0) * self.stp_x              # [V,n_l5]
        self.stp_x = (self.stp_x - u_rel * act.unsqueeze(0)
                      ).clamp(min=0.0)
        self.stp_u = (self.stp_u
                      + u_eff.unsqueeze(0) * (1.0 - self.stp_u)
                      * act.unsqueeze(0)).clamp(max=1.0)
        self.stp_x += (1.0 - self.stp_x) * (dt / tau_d_eff.unsqueeze(0))
        self.stp_u += (c.stp_u0 - self.stp_u) * (dt / c.stp_tau_f)
        # Усталость выхода: 3 подряд спайка → ресурс ~0.1;
        # восстановление ~25-30 мс, замедленное при высокой частоте.
        self.out_fatigue = (self.out_fatigue
                            - 0.4 * (1.0 + self.rate_post)
                            * s.mean(0)).clamp(min=0.02)
        self.out_fatigue += (1.0 - self.out_fatigue) * (
            dt / (c.out_tau * (1.0 + 2.0 * self.rate_post)))

    # ------------------------------------------------------------------
    def set_bg_decision(self, action: Optional[torch.Tensor],
                        n_actions: int = 0) -> None:
        """Решение BG-DDM как one-hot канал в моторный пул."""
        if action is None or self.w_bg is None:
            self._bg_decision = None
            return
        a = int(action.flatten()[0].item()) if torch.is_tensor(action)             else int(action)
        if a < 0:
            self._bg_decision = None
            return
        oh = torch.zeros(self.w_bg.shape[1])
        oh[min(a, self.w_bg.shape[1] - 1)] = 1.0
        self._bg_decision = oh

    def step(self, s_l5: torch.Tensor,
             hipp_v: Optional[torch.Tensor] = None,
             lap_slow: Optional[torch.Tensor] = None,
             dt: float = 1.0, ou_on: bool = True) -> MotorOut:
        """Токен-шаг: sub_ticks мс интеграции → WTA-победитель.

        Выбор = нейрон с максимумом спайков за окно; при нулевом счёте —
        максимально деполяризованный (ближайший к порогу) — та же гонка,
        не декодер-логиты."""
        c = self.cfg
        self.counts = torch.zeros_like(self.v)
        for _ in range(c.sub_ticks):
            self._sub_step(s_l5, hipp_v, lap_slow, dt, ou_on)
        counts = self.counts.mean(0)                     # [V]
        if float(counts.max()) > 0.0:
            token = int(counts.argmax().item())          # WTA по спайкам
        else:
            token = int(self.v.mean(0).argmax().item())  # гонка мембран
        # Адаптация победителя: максимальная деполяризация → максимум
        # Ca²⁺-притока → sAHP-порог растёт и без спайка (повторный
        # выбор того же нейрона сам себя тормозит).
        self.theta[:, token] += c.beta_adapt * 2.0
        vv = self.v.mean(0)
        top2 = vv.topk(min(2, self.vocab)).values
        margin = float(top2[0] - (top2[1] if top2.numel() > 1
                                  else top2[0] - 1.0))
        self._last_counts = counts
        return MotorOut(token=token,
                        spikes=(self.counts > 0).float(),
                        v=self.v, margin=margin,
                        stp_depleted=float(self.stp_x[token].mean()),
                        firing_rate=counts)

    # ------------------------------------------------------------------
    def learn(self, target: int, s_l5: torch.Tensor, f_mod: float,
              plasticity_ok: bool = True, reward: float = 0.0,
              dt: float = 1.0) -> float:
        """Трёхфакторное обновление w_l5 (Законы 4–5).

        err_i = H(i==target)·1 − rate_i: нейрон цели «подталкивается»
        к активности, прочие приглушаются — обучение без градиентов
        и без MLP (Urbanczik-Senn nudging × DA-фактор)."""
        if not plasticity_ok or f_mod <= 0.0:
            return 0.0
        rate = self._last_counts / max(1, self.cfg.sub_ticks)
        err = torch.zeros(self.vocab)
        err[target] = 1.0
        err = err - rate
        x = s_l5.mean(0)                                 # пре-след L5
        self.eligibility = self.eligibility * (1.0 - dt / 1000.0) \
            + torch.outer(err, x)
        dw = self.cfg.eta_motor * f_mod * (
            torch.outer(err, x) + reward * self.eligibility * 0.5)
        self.w_l5 = (self.w_l5 + dw).clamp(-2.0, 2.0)
        self.z_pre = self.z_pre * (1.0 - dt / 20.0) + x
        self.z_post = self.z_post * (1.0 - dt / 20.0) + rate
        return float(dw.abs().mean())
