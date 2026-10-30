"""GenesisOrganism (§III псевдокод): полный цикл 1 мс.

Фазы такта:
  0. Нейромодуляторы: RPE→DA, novelty→NE, внимание→ACh, OX/ADO
     бистабильность wake/sleep, кортизол-режим Φ_CORT, f_mod.
  1. Сенсорика: SDR → TRN-гейт → ThalamusFO (tonic/burst) → L4.
  2. Кора: FF→basal AMPA, FB→apical NMDA (SST/VIP гейт), PV k-WTA,
     HO-трансталамический драйв, GW-бродкаст, трёхфакторное обучение.
  3. Память/мотивация: HPC encode (бодр.) или SWR-реплей (сон),
     NO-поле/2-AG, BG-DDM выбор действия.
  4. Сознание: кандидаты зон → GW-Hopfield → ignition → broadcast,
     MetaHead HOT, AST интроспекция и корректирующий гейн → TRN.
  5. Гомеостаз: MR-σ̂, SOC-регуляция, IP, iSTDP, лактат.

Патологии (§4.1): эпилепсия σ>1.2>100мс, кома σ<0.7>500мс,
метаболический коллапс E<0>50% нейронов, вечный сон/бодрствование,
взрыв/коллапс весов → аварийная остановка.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from ..action.basal_ganglia import BasalGanglia, BGOut
from ..chemistry.modulators import NeuroChemistry
from ..chemistry.agents import ExtendedChemistry
from ..consciousness.metrics import ConsciousnessDashboard
from ..chemistry.volume import VolumeTransmission
from ..config import GenesisConfig
from ..consciousness.attention_schema import AttentionSchema, ASTOut
from ..consciousness.meta_head import MetaHead, MetaOut
from ..consciousness.workspace import GlobalWorkspace, GWOut
from ..cortex.hierarchy import CorticalHierarchy, HierarchyOut
from ..cortex.astrocyte import AstrocyteMicrodomain
from ..homeostasis.controller import HomeostaticController
from ..memory.hippocampus import Hippocampus, HippoOut
from ..memory.laplace import LaplaceBank
from ..thalamus.relay import ThalamicHO, ThalamusFO
from ..thalamus.trn import TRN


@dataclass
class GenesisOutput:
    t: int
    action: torch.Tensor          # выбранное действие BG (-1 = нет)
    gw_content: List[torch.Tensor]
    confidence: float
    introspection: Dict           # AST-отчёт
    sigma: float
    is_awake: bool
    sleep_pressure: float
    energy: float                 # средний ATP
    prediction_errors: Dict[str, float]
    pathology: Optional[str] = None


class GenesisOrganism:
    """Полный организм v10.0 «Genesis»."""

    def __init__(self, cfg: GenesisConfig, batch: int = 1, seed: int = 10):
        self.cfg = cfg
        self.batch = batch
        torch.manual_seed(seed)
        self.chem = NeuroChemistry(cfg.chem)
        self.hier = CorticalHierarchy(cfg, batch)
        n_total = sum(z.n_neurons for z in cfg.zones)
        self.fo = ThalamusFO(cfg.input_dim)
        self.trn = TRN(len(cfg.zones))
        l5 = lambda n: n - 2 * n // 3  # ширина L5-подмножества колонки
        self.ho = {cfg.zones[i + 1].name:
                   ThalamicHO(l5(cfg.zones[i].n_neurons),
                              cfg.zones[i].n_neurons)
                   for i in range(len(cfg.zones) - 1)}
        self.hipp = Hippocampus(cfg.hippo, seed)
        self.bg = BasalGanglia(cfg.bg, seed)
        self.gw = GlobalWorkspace(cfg.gw)
        self.meta = MetaHead(cfg.meta, cfg.gw.d_gw, seed)
        self.ast = AttentionSchema(cfg.ast, seed)
        self.volume = VolumeTransmission(n_total, cfg.chem.no_tau,
                                         cfg.chem.no_d, cfg.chem.no_radius,
                                         cfg.chem.h2s_tau)
        self.homeo = HomeostaticController(cfg.homeo)
        self.xchem = ExtendedChemistry()
        from ..memory.working import SilentWorkingMemory
        self.wm = SilentWorkingMemory(cfg.zones[-1].n_neurons)
        from ..core.interoception import InteroceptiveField
        self.intero = InteroceptiveField([z.name for z in cfg.zones])
        self.dash = ConsciousnessDashboard()
        # REGENESIS Закон 4: память и модуляторы — в физике, не в векторе
        # Астроциты микродоменов: stp_gain синапсов + θ_Ca плато дендрита
        self.astro = [AstrocyteMicrodomain(z.n_neurons) for z in cfg.zones]
        # Лапласов спектр: медленные шкалы → возбудимость L5
        self.lap = LaplaceBank(cfg.input_dim)
        # Гиппокамп CA1→L5: NMDA+Mg²⁺-гейт (прайминг, не спайк)
        l5w = lambda n: n - 2 * n // 3
        self.w_hpc = {z.name: torch.randn(l5w(z.n_neurons),
                                          cfg.hippo.n_ec) * 0.2
                      for z in cfg.zones}
        self._prev_ca1: Optional[torch.Tensor] = None
        self.plasticity_blocked = False
        self.t = 0
        self.pathology: Optional[str] = None
        self._sigma_hi_ms = 0
        self._sigma_lo_ms = 0
        self._prev_spikes: Optional[torch.Tensor] = None
        self.sigma = 1.0
        self._last_meta: Optional[MetaOut] = None
        self._last_ast: Optional[ASTOut] = None

    def reset_state(self, batch: int) -> None:
        self.batch = batch
        self.hier.reset_state(batch)
        self.lap.reset_state(batch)
        for a in self.astro:
            a.reset_state(batch)
        self._prev_spikes = None
        self._prev_ca1 = None

    # ------------------------------------------------------------------
    def _check_pathology(self, spikes_all: torch.Tensor) -> Optional[str]:
        p = self.cfg.patho
        if self.sigma > p.sigma_epilepsy:
            self._sigma_hi_ms += 1
            if self._sigma_hi_ms > p.sigma_epilepsy_ms:
                return "epilepsy"
        else:
            self._sigma_hi_ms = 0
        if self.sigma < p.sigma_coma:
            self._sigma_lo_ms += 1
            if self._sigma_lo_ms > p.sigma_coma_ms:
                return "coma"
        else:
            self._sigma_lo_ms = 0
        # Метаболический коллапс
        e = torch.cat([c.pyr.energy.flatten() for c in self.hier.cols])
        if float((e < 0.0).float().mean()) > p.energy_collapse_frac:
            return "metabolic_collapse"
        if self.chem.ticks_asleep > p.eternal_sleep_ticks:
            return "eternal_sleep"
        if self.chem.ticks_awake > p.eternal_wake_ticks:
            return "eternal_wake"
        # Веса: взрыв или коллапс
        for col in self.hier.cols:
            w_max = max(float(col.w_ff.w.abs().max()),
                        float(col.w_fb.w.abs().max()))
            if w_max > p.w_explode:
                return "weight_explosion"
            if 0.0 < w_max < p.w_collapse:
                return "weight_collapse"
        return None

    # ------------------------------------------------------------------
    def step(self, x: torch.Tensor, reward: float = 0.0,
             learn: bool = True, drive: float = 0.6,
             stress: float = 0.0) -> GenesisOutput:
        """Один такт dt=1 мс. x [B,input_dim]."""
        cfg = self.cfg
        s = self.chem.s
        self.t += 1

        # ---- Фаза 0: нейромодуляторы ----
        activity = float(self._prev_spikes.mean()) \
            if self._prev_spikes is not None else 0.05
        self.chem.update(activity=activity, rpe=reward,
                         novelty=getattr(self, "_last_novelty", 0.0),
                         attention=float(self.ast.gain.mean())
                         if hasattr(self, "ast") else 0.5,
                         drive=drive, stress=stress,
                         spikes=activity)
        e_mean = float(torch.cat(
            [c.pyr.energy.flatten() for c in self.hier.cols]).mean()) \
            if all(hasattr(c.pyr, "energy") for c in self.hier.cols) \
            else 1.0
        energy_ok = e_mean > cfg.chem.theta_energy
        # Закон 5: аденилатный долг выше порога → w_fast не обновляются
        # (пластичность — метаболически дорогой процесс).
        debt = 1.0 - e_mean
        self.plasticity_blocked = debt > cfg.chem.atp_debt_block
        f_mod = self.chem.f_mod(energy_ok)
        if self.plasticity_blocked:
            f_mod = 0.0
        awake = self.chem.is_awake

        # ---- Фаза 1: сенсорика → таламус ----
        if awake:
            sector_req = torch.ones(len(cfg.zones)) * 0.5
            trn_gate = self.trn.step(sector_req,
                                   self.ast.gain[:len(cfg.zones)]
                                   if self._last_ast else None)
            fo_out = self.fo.step(x, s.ach, s.ne)
        else:
            fo_out = torch.zeros_like(x)

        # ---- Фаза 4a (ранняя): GW-кандидаты от прошлого такта ----
        gw_b = self._gw_broadcast if hasattr(self, "_gw_broadcast") else None

        # ---- Фаза 3a: BG-дизингибиция HO + контекст PFC ----
        pfc_s = self._prev_pfc if hasattr(self, "_prev_pfc") else \
            torch.zeros(self.batch, cfg.zones[-1].n_neurons)
        bg_out: BGOut = self.bg.step(pfc_s, s.ox, s.ado, s.ht)
        ho_drive: Dict[str, torch.Tensor] = {}
        l5_prev = self._prev_l5 if hasattr(self, "_prev_l5") else {}
        for zname, ho in self.ho.items():
            src = cfg.zones[cfg.zones.index(cfg.zone(zname)) - 1].name \
                if hasattr(cfg.zones, "index") else None
            src_name = [z.name for z in cfg.zones][
                [z.name for z in cfg.zones].index(zname) - 1]
            drive_in = l5_prev.get(src_name, torch.zeros(
                self.batch, ho.n_in))
            _ = src  # src-зона уже найдена ниже
            # Базовая дизингибиция BG держит HO приоткрытым; решение —
            # полное открытие выбранного канала.
            ho_drive[zname] = ho.step(
                drive_in, 0.65 + 0.35 * float(bg_out.decided))

        # ---- Фаза 2: кортикальные зоны ----
        vip_mod = {z.name: float(s.ht * 0.3 + 0.1) for z in cfg.zones}
        # Расширенный химический пул: медленные агенты и их эффекты
        self.xchem.update(spikes=activity, stress=stress, arousal=float(awake),
                          social=float(drive), threat=stress,
                          damage=getattr(self, '_last_mismatch', 0.0))
        mod_ctx = {"ach": s.ach, "histamine": self.xchem.l.histamine,
                   "nmda_gain": self.xchem.nmda_gain(),
                   "p_rel": self.xchem.p_release_scale(),
                   "vth_shift": self.xchem.vth_shift(),
                   "gaba_tonic": self.xchem.gaba_tonic()}
        # Закон 4: Лапласовы медленные шкалы → адаптация порога L5
        lap_slow = self.lap.step(x).mean(0)
        lap_scalar = float(self.lap.slow_profile().mean())
        # Астроциты прошлого такта: stp_gain синапсов + θ_Ca плато
        for col, ast in zip(self.hier.cols, self.astro):
            col._mod_ctx = mod_ctx
            col._hipp_in = self._prev_ca1 @ self.w_hpc[col.name].t() \
                if self._prev_ca1 is not None else None
            if self.t > 1:
                glio = ast.step(col._last_s if hasattr(col, "_last_s")
                                else torch.zeros(self.batch,
                                                 col.zcfg.n_neurons),
                                ne=s.ne)
                col.w_ff.stp_gain = glio.mean(0).unsqueeze(1)  # [N,1]
                col.dendrite.theta_ca = -25.0 + ast.theta_ca_shift()
            for w in (col.w_ff, col.w_fb, col.w_rec):
                w.set_metaplastic(gc=s.cort, e2=self.xchem.l.e2,
                                  h2s=self.xchem.l.h2s if hasattr(self.xchem.l,'h2s') else 0.2,
                                  zn=self.xchem.l.zn, mg=1.0)
        hout: HierarchyOut = self.hier.step(
            fo_out, gw_b, ho_drive, f_mod, vip_mod, learn=learn)
        # Лапласовы медленные шкалы → adapt L5: приращение порога на
        # спайк масштабируется длинным контекстом — клетка становится
        # более/менее адаптируемой в зависимости от истории входа.
        lap_gain = 1.0 + (lap_scalar - 0.5) * 0.8
        for col in self.hier.cols:
            s_l5 = col._last_s[:, col.l5_idx] \
                if hasattr(col, "_last_s") else 0.0
            col.pyr.theta[:, col.l5_idx] += \
                s_l5 * (lap_gain - 1.0) * cfg.neuron.beta_adapt * cfg.dt
        self._prev_l5 = hout.l5
        self._prev_pfc = hout.zones[cfg.zones[-1].name].s
        self._prev_spikes = hout.spikes_all

        # ---- Фаза 3b: гиппокамп, объёмная передача ----
        ec_in = torch.cat([
            torch.nn.functional.pad(hout.zones["SENS"].s,
                                    (0, max(0, cfg.hippo.n_ec
                                            - hout.zones["SENS"].s.shape[1])))
            [:, :cfg.hippo.n_ec],
        ], dim=1) if "SENS" in hout.zones else torch.zeros(
            self.batch, cfg.hippo.n_ec)
        err_mean = float(hout.spikes_all.abs().mean())
        unc = self._last_meta.epistemic if self._last_meta else 0.5
        hippo_out: HippoOut = self.hipp.step(ec_in, s.ach, s.ado,
                                           learn=learn and awake)
        self.hipp.store_episode(ec_in, hippo_out.dg, hippo_out.ca3,
                                err_mean, unc,
                                da_tag=hippo_out.vta_da_burst)
        self._prev_ca1 = hippo_out.ca1_out.detach()
        self._last_lap_slow = lap_slow
        if not awake:  # SWR-реплей
            def cortical_write(ec, err):
                for col in self.hier.cols:
                    for w in (col.w_ff, col.w_rec):
                        w.w += cfg.synapse.eta * 0.05 * err * w.w.sign() \
                            * w.mask
            replayed = self.hipp.swr_replay(cortical_write)
            hippo_out.replayed = replayed
        no_field = self.volume.step(
            torch.cat([hout.zones[z.name].ca for z in cfg.zones], dim=1),
            hout.spikes_all)
        ca_mean = float(torch.cat(
            [hout.zones[z.name].ca.mean(0) for z in cfg.zones]).mean())
        self.chem.ecb_update(ca_mean)
        # Нейростероидный/NO/2-AG вход нейронам
        for col in self.hier.cols:
            col.pyr.allo = (s.cort * 0.3 * torch.ones_like(col.pyr.allo))
            col.pyr.ado_i = s.ado * torch.ones_like(col.pyr.ado_i)

        # ---- Фаза 4: сознание ----
        candidates = []
        for z in cfg.zones:
            srate = hout.zones[z.name].s.mean(0)
            cand = torch.nn.functional.pad(
                srate, (0, max(0, cfg.gw.d_gw - srate.numel())))[
                :cfg.gw.d_gw]
            candidates.append(cand)
        candidates.append(torch.nn.functional.pad(
            hippo_out.ca1_out.mean(0)[:cfg.gw.d_gw],
            (0, max(0, cfg.gw.d_gw - hippo_out.ca1_out.shape[1]))))
        gw_out: GWOut = self.gw.compete(candidates, s.ne)
        self._gw_broadcast = gw_out.broadcast
        meta_out = self.meta.step(gw_out.broadcast, gw_out.content,
                                  err_mean, reward)
        self._last_meta, self._last_novelty = meta_out, meta_out.novelty
        ast_out = self.ast.step(meta_out.novelty, err_mean, drive,
                                actual_gain=float(trn_gate.mean())
                                if awake else 0.0,
                                actual_focus=gw_out.winner_zone
                                if gw_out.winner_zone >= 0 else 0)
        # Интероцепция: сеть читает собственное состояние — метапознание
        zone_states = {z.name: torch.tensor([
            float(hout.zones[z.name].v_s.mean()) / 100.0 + 0.5,
            float(col.pyr.energy.mean()),
            float(hout.zones[z.name].ca.mean()),
            float(hout.zones[z.name].s.mean())])
            for z, col in zip(cfg.zones, self.hier.cols)}
        intero_out = self.intero.read(zone_states)
        ast_out.report["self_state"] = intero_out.summary
        ast_out.report["intero_activity"] = float(intero_out.state_vec.mean())
        self._last_intero = intero_out
        self._last_ast = ast_out
        self._last_mismatch = hippo_out.mismatch

        # ---- Фаза 5: гомеостаз ----
        if self._prev_spikes is not None:
            self.sigma = self.homeo.monitor.push(
                hout.spikes_all.mean(0),
                self._prev_spikes.mean(0))
        hom_rep = self.homeo.regulate(self.hier, self.sigma)
        # Рабочая память PFC: спайк-паттерн → STP-фацилитация (без огня)
        if awake:
            self.wm.write(self._prev_pfc > 0, strength=0.1)
        self.wm.tick(dt=1.0, active_hold=awake)
        # Нейромод-эффекты: novelty → NE/DA-бёрсты от CA1
        s.ne += hippo_out.lc_ne_burst * 0.1
        s.da += hippo_out.vta_da_burst * 0.05

        self.pathology = self._check_pathology(hout.spikes_all)
        errors = {z.name: float(hout.errors[z.name].abs().mean())
                  for z in cfg.zones}
        # Дашборд сознания: PCI/R_bc/γ_ign/ECE/каузальная плотность
        r_bc = float((gw_out.broadcast.abs().mean() > 1e-4)
                     * sum(float(hout.errors[z.name].abs().mean() > 1e-6)
                           for z in cfg.zones) / max(1, len(cfg.zones)))
        self.dash.push(hout.spikes_all.flatten(),
                       float(gw_out.ignition),
                       float(gw_out.broadcast.abs().mean() > 1e-6),
                       meta_out.confidence, float(1.0 - min(1.0, err_mean)),
                       float(1.0 - ast_out.expected_accuracy), r_bc)
        self._last_dash = self.dash.report(
            ffr=float(sum(float(c.w_fb.w.abs().mean()) for c in self.hier.cols)
                      / max(1e-9, sum(float(c.w_ff.w.abs().mean())
                                      for c in self.hier.cols))))
        return GenesisOutput(
            t=self.t, action=bg_out.action,
            gw_content=gw_out.content, confidence=meta_out.confidence,
            introspection=ast_out.report, sigma=self.sigma,
            is_awake=awake, sleep_pressure=s.ado,
            energy=float(torch.cat(
                [c.pyr.energy.flatten() for c in self.hier.cols]).mean()),
            prediction_errors=errors, pathology=self.pathology)

    # ------------------------------------------------------------------
    def run_episode(self, xs: List[torch.Tensor], learn: bool = True,
                    drive: float = 0.6) -> List[GenesisOutput]:
        outs = []
        for x in xs:
            outs.append(self.step(x, learn=learn, drive=drive))
            if self.pathology:
                break
        return outs

    def sleep_cycle(self, ticks: int = 200) -> Dict:
        """Принудительный сон: ACh↓, Ado↑ → SWR-реплей + консолидация."""
        self.chem.is_awake = False
        self.chem.s.ado = 0.8
        self.chem.s.ach = 0.1
        n_replayed = 0
        for i in range(ticks):
            self.step(torch.zeros(self.batch, self.cfg.input_dim),
                      learn=True)
            # Сонные веретёна: TRN-осцилляции ~10 Гц синхронизируют
            # таламо-кортикальную петлю с SWR-реплеем гиппокампа
            _ = self.fo.spindle(i)
            n_replayed += len(getattr(self.hipp, "_dbg_replayed", []))
        self.hier.consolidate(0.5)
        self.chem.s.ado = 0.3
        self.chem.s.ox = 0.6
        self.chem.is_awake = True
        return {"ticks": ticks, "sigma": self.sigma,
                "episodes": len(self.hipp.episodes)}
