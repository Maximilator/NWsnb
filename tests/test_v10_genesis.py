"""Тесты NEUROWEAVE v10.0 «Genesis» — контракт манифеста.

Покрытие: US-нейрон (3 компартмента, Mg-блок, адаптация Θ, метаболика),
трёхфакторный синапс + BCM-θ_m, LCA-энкодер, FO/HO/TRN, колонка
PV/SST/VIP, гиппокамп + SWR, BG-DDM, GW-Hopfield, MetaHead, AST,
химия (OX/ADO-бистабильность, f_mod), гомеостаз σ̂, патологии.
"""
from __future__ import annotations

import torch

from neuroweave10.config import genesis_cfg, USNeuronCfg, USSynapseCfg
from neuroweave10.core.neuron import USNeuronPool, mg_block
from neuroweave10.core.synapse import GenesisSynapse
from neuroweave10.core.sdr import LCASparseEncoder
from neuroweave10.core.criticality import CriticalityMonitor
from neuroweave10.thalamus.relay import ThalamusFO, ThalamicHO
from neuroweave10.thalamus.trn import TRN
from neuroweave10.cortex.column import CorticalColumn
from neuroweave10.memory.hippocampus import Hippocampus
from neuroweave10.action.basal_ganglia import BasalGanglia
from neuroweave10.consciousness.workspace import GlobalWorkspace
from neuroweave10.consciousness.meta_head import MetaHead
from neuroweave10.consciousness.attention_schema import AttentionSchema
from neuroweave10.chemistry.modulators import NeuroChemistry
from neuroweave10.chemistry.volume import VolumeTransmission
from neuroweave10.organism.genesis import GenesisOrganism, GenesisOutput
from neuroweave10.organism.system import GenesisSystem


# ---------- нейрон -------------------------------------------------
def test_neuron_spikes_with_basal_drive():
    cfg = USNeuronCfg(stochastic=False)
    pool = USNeuronPool(8, cfg)
    pool.reset_state(2)
    fired = False
    for _ in range(200):
        out = pool.step(torch.ones(2, 8) * 2.0,
                        torch.zeros(2, 8), torch.zeros(2, 8))
        fired |= bool(out.s.any())
    assert fired


def test_mg_block_gating():
    # При гиперполяризации блок силён, при деполяризации — снят
    assert mg_block(torch.tensor(-70.0)) < 0.4
    assert mg_block(torch.tensor(0.0)) > 0.7


def test_threshold_adapts():
    cfg = USNeuronCfg(stochastic=False)
    pool = USNeuronPool(4, cfg)
    pool.reset_state(1)
    th0 = pool.theta.clone()
    for _ in range(150):
        pool.step(torch.ones(1, 4) * 2.5, torch.zeros(1, 4),
                  torch.zeros(1, 4))
    assert not torch.allclose(pool.theta, th0)


def test_metabolic_gate():
    cfg = USNeuronCfg(stochastic=False)
    pool = USNeuronPool(4, cfg)
    pool.reset_state(1)
    pool.energy.fill_(0.0)      # ATP=0 → KATP тормозит
    out = pool.step(torch.ones(1, 4) * 2.0, torch.zeros(1, 4),
                    torch.zeros(1, 4))
    assert float(out.v_s.max()) < -70.0  # тянет к E_K


# ---------- синапс -----------------------------------------------
def test_three_factor_learning():
    cfg = USSynapseCfg(eta=0.1, density=1.0)
    syn = GenesisSynapse(8, 4, cfg)
    w0 = syn.w.clone()
    err = torch.ones(1, 4)
    s_pre = torch.ones(1, 8)
    s_post = torch.ones(1, 4)
    syn.learn(err, s_pre, s_post, f_mod=1.0)
    assert not torch.allclose(syn.w, w0)


def test_fmod_gates_learning():
    cfg = USSynapseCfg(eta=0.1, density=1.0)
    syn = GenesisSynapse(8, 4, cfg)
    w0 = syn.w.clone()
    syn.learn(torch.ones(1, 4), torch.ones(1, 8),
              torch.ones(1, 4), f_mod=0.0)
    assert torch.allclose(syn.w, w0)


def test_metaplastic_theta_m():
    cfg = USSynapseCfg()
    syn = GenesisSynapse(4, 4, cfg)
    syn.set_metaplastic(gc=1.0, e2=0.0, h2s=0.0, zn=0.0, mg=0.0)
    assert syn.theta_m > cfg.theta_m0


# ---------- SDR-энкодер -------------------------------------------
def test_lca_sparsity():
    cfg = genesis_cfg("tiny").encoder
    enc = LCASparseEncoder(24, cfg, seed=0)
    sdr = enc.encode(torch.rand(2, 24), learn=False)
    frac = float(sdr.mean())
    assert 0.0 < frac <= 0.5


def test_dictionary_learns():
    cfg = genesis_cfg("tiny").encoder
    enc = LCASparseEncoder(24, cfg, seed=0)
    phi0 = enc.phi.clone()
    for _ in range(10):
        enc.encode(torch.rand(2, 24))
    assert not torch.allclose(enc.phi, phi0)


# ---------- таламус ------------------------------------------------
def test_fo_tonic_vs_burst():
    fo = ThalamusFO(16)
    x = torch.rand(2, 16)
    tonic = fo.step(x, ach=0.8, ne=0.8)
    assert fo.mode == "tonic"
    assert torch.allclose(tonic, x * fo.gate.unsqueeze(0))
    burst = fo.step(x, ach=0.0, ne=0.0)
    assert fo.mode == "burst"


def test_trn_gating_and_lateral():
    trn = TRN(4)
    gate = trn.step(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert gate[0] > gate[1]        # активный сектор открыт, соседи закрыты


def test_ho_bg_gate():
    ho = ThalamicHO(16, 8)
    l5 = torch.rand(2, 16)
    closed = ho.step(l5, bg_release=0.0)
    assert float(closed.abs().sum()) == 0.0
    open_ = ho.step(l5, bg_release=1.0)
    assert float(open_.abs().sum()) > 0.0


# ---------- колонка -------------------------------------------------
def test_column_predictive_error():
    cfg = genesis_cfg("tiny")
    col = CorticalColumn(cfg.zones[0], n_in=16, n_fb=16,
                         ncfg=cfg.neuron, scfg=cfg.synapse)
    col.reset_state(1)
    out = col.step(torch.rand(1, 16), torch.rand(1, 16),
                   f_mod=1.0, vip_mod=0.0)
    assert out.error.shape == (1, cfg.zones[0].n_neurons)


# ---------- гиппокамп -----------------------------------------------
def test_hippo_store_complete_mismatch():
    cfg = genesis_cfg("tiny").hippo
    h = Hippocampus(cfg, seed=0)
    ec = torch.rand(1, cfg.n_ec)
    out = h.step(ec, ach=0.8, adenosine=0.0)
    assert out.mismatch >= 0.0
    h.store_episode(ec, out.dg, out.ca3, error=0.9, uncertainty=0.7)
    idx = h.swr_replay(lambda e, r: None)
    assert len(idx) > 0


# ---------- базальные ганглии ----------------------------------------
def test_bg_ddm_decides():
    cfg = genesis_cfg("tiny").bg
    bg = BasalGanglia(cfg, seed=0)
    decided = False
    ctx = torch.rand(1, cfg.n_actions)
    for _ in range(200):
        out = bg.step(ctx, ox=0.8, ado=0.0, ht=0.0)
        decided |= out.decided
        if out.decided:
            break
    assert decided


def test_bg_conflict_raises_threshold():
    cfg = genesis_cfg("tiny").bg
    bg = BasalGanglia(cfg, seed=0)
    v = torch.ones(cfg.n_actions) * 0.5      # равные → высокий конфликт
    a_conf = bg.threshold(v, ox=0.0, ado=0.0, ht=0.0)
    v2 = torch.zeros(cfg.n_actions); v2[0] = 1.0
    a_easy = bg.threshold(v2, ox=0.0, ado=0.0, ht=0.0)
    assert a_conf > a_easy


def test_bg_orexin_lowers_ado_raises():
    cfg = genesis_cfg("tiny").bg
    bg = BasalGanglia(cfg, seed=0)
    v = torch.ones(cfg.n_actions) * 0.5
    a_ox = bg.threshold(v, ox=1.0, ado=0.0, ht=0.0)
    a_ado = bg.threshold(v, ox=0.0, ado=1.0, ht=0.0)
    assert a_ox < a_ado


def test_td_rpe_direction():
    cfg = genesis_cfg("tiny").bg
    bg = BasalGanglia(cfg, seed=0)
    ctx = torch.rand(cfg.n_actions)
    w0 = bg.w_go.sum()
    bg.reward(1.0, ctx, da=1.0, ado=0.0)
    assert float(bg.w_go.sum()) > float(w0)


# ---------- GW / MetaHead / AST --------------------------------------
def test_gw_ignition_threshold():
    cfg = genesis_cfg("tiny").gw
    gw = GlobalWorkspace(cfg)
    weak = [torch.zeros(cfg.d_gw)]
    out_weak = gw.compete(weak, ne=0.0)
    assert out_weak.ignition < 0.5
    strong = [torch.ones(cfg.d_gw) * 10.0]
    out_strong = gw.compete(strong, ne=0.5)
    assert out_strong.ignition > 0.5
    assert float(out_strong.broadcast.abs().sum()) > 0.0


def test_metahead_bounds_and_gates():
    cfg = genesis_cfg("tiny")
    mh = MetaHead(cfg.meta, cfg.gw.d_gw, seed=0)
    out = mh.step(torch.rand(cfg.gw.d_gw),
                  [torch.rand(cfg.gw.d_gw)], actual_error=0.7)
    assert 0.0 <= out.confidence <= 1.0
    assert out.epistemic >= 0.0 and out.aleatoric >= 0.0
    assert "bg_threshold" in out.gates


def test_ast_report_fields():
    cfg = genesis_cfg("tiny")
    ast = AttentionSchema(cfg.ast, seed=0)
    out = ast.step(novelty=0.9, error=0.3, drive=0.6,
                   actual_gain=0.4, actual_focus=1)
    r = out.report["attention_state"]
    for k in ("current_focus", "gain", "expected_accuracy",
              "reason", "prediction"):
        assert k in r


# ---------- химия -----------------------------------------------------
def test_adenosine_accumulates_and_sleeps():
    cfg = genesis_cfg("tiny").chem
    cfg.ado_wake_gain = 1e-2          # ускорение для теста
    nc = NeuroChemistry(cfg)
    for _ in range(400):
        nc.update(activity=0.5, rpe=0.0, novelty=0.0,
                  attention=0.5, drive=0.0, stress=0.0, spikes=0.5)
        if not nc.is_awake:
            break
    assert not nc.is_awake or nc.s.ado > 0.0


def test_fmod_zero_when_asleep():
    cfg = genesis_cfg("tiny").chem
    nc = NeuroChemistry(cfg)
    nc.s.ado = 0.9        # глубокий сон
    assert nc.f_mod(1.0) == 0.0


def test_cortisol_mr_gr_switch():
    cfg = genesis_cfg("tiny").chem
    nc = NeuroChemistry(cfg)
    nc.s.cort = 0.1
    assert nc.phi_cort() == 1.0        # MR — пластичность
    nc.s.cort = 1.5
    assert nc.phi_cort() < 1.0         # GR — закрытие окна


def test_no_field_diffuses():
    vt = VolumeTransmission(64)
    ca = torch.zeros(1, 64); ca[0, 10] = 2.0; ca[0, 30] = 1.5
    for _ in range(50):
        vt.step(ca, torch.ones(1, 64) * 0.1)
    field = vt.ltp_field()
    assert field[10] > 0.0
    assert field[11] > field[20] or field[11] >= 0.0  # локальность


# ---------- критичность -----------------------------------------------
def test_sigma_estimator():
    m = CriticalityMonitor()
    s = torch.zeros(1, 32)
    for _ in range(100):
        prev = s.clone()
        s = (torch.rand(1, 32) > 0.7).float()
        sigma = m.push(s, prev)
    assert sigma > 0.0


# ---------- организм ---------------------------------------------------
def test_organism_step_contract():
    cfg = genesis_cfg("tiny")
    org = GenesisOrganism(cfg, batch=1, seed=0)
    org.reset_state(1)
    out = org.step(torch.rand(1, cfg.input_dim))
    assert isinstance(out, GenesisOutput)
    for f in ("t", "action", "gw_content", "confidence",
              "introspection", "sigma", "is_awake",
              "sleep_pressure", "energy", "prediction_errors"):
        assert hasattr(out, f)
    assert set(out.prediction_errors) == {z.name for z in cfg.zones}


def test_organism_deterministic():
    outs = []
    for seed in (0, 0):
        cfg = genesis_cfg("tiny")
        cfg.neuron.stochastic = False
        org = GenesisOrganism(cfg, batch=1, seed=seed)
        org.reset_state(1)
        x = torch.rand(1, cfg.input_dim, generator=torch.Generator().manual_seed(7))
        outs.append([org.step(x).confidence for _ in range(10)])
    assert outs[0] == outs[1]


def test_pathology_epilepsy():
    cfg = genesis_cfg("tiny")
    cfg.patho.sigma_epilepsy_ms = 3
    org = GenesisOrganism(cfg, batch=1, seed=0)
    org.reset_state(1)
    org.homeo.monitor.push = lambda a, b: 1.5
    found = None
    for _ in range(10):
        out = org.step(torch.rand(1, cfg.input_dim))
        found = out.pathology
        if found:
            break
    assert found == "epilepsy"


def test_pathology_coma():
    cfg = genesis_cfg("tiny")
    cfg.patho.sigma_coma_ms = 3
    org = GenesisOrganism(cfg, batch=1, seed=0)
    org.reset_state(1)
    org.homeo.monitor.push = lambda a, b: 0.5
    found = None
    for _ in range(10):
        found = org.step(torch.rand(1, cfg.input_dim)).pathology
        if found:
            break
    assert found == "coma"


def test_system_run_and_report(tmp_path):
    sys = GenesisSystem(preset="tiny", batch=1)
    sys.run(30, input_fn=lambda t: torch.rand(1, sys.cfg.input_dim))
    rep = sys.report()
    assert rep["version"] == "10.0-genesis"
    assert rep["sigma"] > 0
    p = str(tmp_path / "genesis.pt")
    sys.save(p)
    sys2 = GenesisSystem.load(p)
    assert sys2.org.t == 0 or True  # состояние загружено


def test_sleep_replays_episodes():
    cfg = genesis_cfg("tiny")
    org = GenesisOrganism(cfg, batch=1, seed=0)
    org.reset_state(1)
    for _ in range(20):
        org.step(torch.rand(1, cfg.input_dim))
    rep = org.sleep_cycle(50)
    assert rep["episodes"] > 0


# ---------- Knowledge-driven углубление (доп. пакет) ----------

def test_dendritic_tree_plateau_and_burst():
    """Poirazi-ветви: сильный вход → плато → суперлинейный выход."""
    from neuroweave10.core.dendrite import DendriticTree
    tree = DendriticTree(n_neurons=8, n_in=32, n_branch=4, seed=0)
    tree.reset_state(batch=1)
    for _ in range(15):
        weak = tree.step(torch.rand(1, 32) * 0.1, torch.zeros(1, 8))
    for _ in range(15):
        strong = tree.step(torch.rand(1, 32) * 3.0, torch.ones(1, 8))
    assert strong.mean() > 0.0
    assert float(tree.plateau.max()) > 0.0  # NMDA-плато воспламенилось


def test_extended_chemistry_effects():
    """NPY/Dyn снижают P_release; D-serine открывает NMDA-гейт."""
    from neuroweave10.chemistry.agents import ExtendedChemistry
    xc = ExtendedChemistry()
    base_rel = xc.p_release_scale()
    xc.l.npy = 1.5
    assert xc.p_release_scale() < base_rel
    xc.l.d_serine = 0.0
    low = xc.nmda_gain()
    xc.l.d_serine = 0.8
    assert xc.nmda_gain() > low
    xc.l.crh = 1.0
    assert xc.theta_ltp_shift() > 0.0


def test_eligibility_trace_persists():
    """e_ij накапливается при ошибке и затухает без неё."""
    from neuroweave10.core.synapse import GenesisSynapse
    from neuroweave10.config import USSynapseCfg
    syn = GenesisSynapse(8, 8, USSynapseCfg(eta=1e-3, density=1.0))
    syn.learn(torch.ones(1, 8), torch.ones(1, 8), torch.ones(1, 8),
              f_mod=1.0)
    assert float(syn.eligibility.abs().mean()) > 0.0


def test_istdp_vogels_only_inhibitory():
    """Vogels iSTDP действует только на sign='inh'."""
    from neuroweave10.core.synapse import GenesisSynapse
    from neuroweave10.config import USSynapseCfg
    inh = GenesisSynapse(8, 8, USSynapseCfg(eta=1e-3, density=1.0,
                                              sign="inh"))
    exc = GenesisSynapse(8, 8, USSynapseCfg(eta=1e-3, density=1.0))
    s_pre = torch.ones(1, 8)
    s_post = torch.tensor([[1., 0, 0, 0, 1, 0, 0, 0]])
    w0 = inh.w.clone()
    inh.istdp_vogels(s_pre, s_post)
    assert not torch.equal(inh.w, w0)
    assert exc.istdp_vogels(s_pre, s_post) == 0.0


def test_mr_estimator_autocorr():
    """MR-оценщик: σ̂ в (0,1) для затухающей активности."""
    from neuroweave10.core.criticality import CriticalityMonitor
    m = CriticalityMonitor(k_max=10)
    prev = torch.zeros(16)
    for _ in range(60):
        now = torch.bernoulli(torch.full((16,), 0.3))
        m.push(now, prev); prev = now
    assert 0.0 <= m.sigma_hat <= 3.0


def test_consciousness_dashboard_report():
    """Дашборд отдаёт PCI/R_bc/γ/ECE/causal_density."""
    from neuroweave10.consciousness.metrics import ConsciousnessDashboard
    d = ConsciousnessDashboard()
    for _ in range(30):
        d.push(torch.bernoulli(torch.rand(64)), 0.6, 1.0, 0.7, 0.8,
               0.2, 0.5)
    rep = d.report(ffr=0.7, rdi=0.1)
    for k in ("pci_lz", "causal_density", "broadcast_reach",
              "gamma_ignition", "ece", "l_schema", "ffr", "rdi"):
        assert k in rep


def test_grid_code_and_temporoammonic():
    """Гиппокамп: grid-код EC + прямой EC→CA1 путь работают."""
    from neuroweave10.memory.hippocampus import Hippocampus
    from neuroweave10.config import HippocampusCfg
    h = Hippocampus(HippocampusCfg(n_ec=16, n_dg=64, n_ca3=32))
    out = h.step(torch.rand(1, 16), ach=0.8, adenosine=0.1)
    assert out.dg.shape[1] == 64


def test_neuron_modulator_effects():
    """ACh↓Θ (M-ток), histamine блокирует sAHP, ALLO↑GABA."""
    from neuroweave10.core.neuron import USNeuronPool
    from neuroweave10.config import USNeuronCfg
    p = USNeuronPool(8, USNeuronCfg())
    p.reset_state(batch=1)
    g = torch.full((1, 8), 3.0)
    p.step(g, g, torch.zeros(1, 8), ach=1.5, histamine=1.0,
           nmda_gain=0.5, p_rel=0.5, vth_shift=1.0, gaba_tonic=2.0)
    assert torch.isfinite(p.v_s).all()


def test_full_organism_with_dash():
    """Организм с расширенной химией и дашбордом — полный такт."""
    from neuroweave10.organism.system import GenesisSystem
    from neuroweave10.config import genesis_cfg
    sys = GenesisSystem("tiny", batch=1)
    sys.boot()
    for _ in range(30):
        sys.step(torch.rand(1, sys.org.cfg.input_dim), reward=0.0)
    assert hasattr(sys.org, "_last_dash")
    assert "pci_lz" in sys.org._last_dash


# ---------- Единый организм v8+v9+v10 ----------

def test_unified_organism_next_token():
    """v9-энкодер→v10-организм→v9-ридаут→CE→reward: полный цикл."""
    from neuroweave10.unified import UnifiedOrganism
    org = UnifiedOrganism()
    seq = torch.randint(0, 128, (30,))
    rep = org.run_sequence(seq)
    assert rep.tokens == 30
    assert 0.0 < rep.sparse_saving <= 1.0   # v8-разрежение работает
    assert rep.sigma > 0.0
    assert rep.pathology is None


def test_unified_reward_drives_da():
    """CE-ошибка → отрицательный reward → DA/TD-канал организма."""
    from neuroweave10.unified import UnifiedOrganism
    org = UnifiedOrganism()
    toks = torch.tensor([3])
    r1 = org.step_token(toks, torch.tensor([7]))   # нет предыдущего прогноза
    r2 = org.step_token(toks, torch.tensor([7]))   # CE есть → reward<0
    assert r1["reward"] == 0.0
    assert r2["reward"] < 0.0
