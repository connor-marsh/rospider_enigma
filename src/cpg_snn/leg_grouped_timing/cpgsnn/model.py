"""
The networks: the surrogate spike function, DenseSNN (the fully connected baseline),
TimingGroupedSNN (CPG -> timing layer or direct CPG drive -> disconnected sub-networks),
parameter blocks, and the single-step ONNX wrappers.
"""

import math

import numpy as np
import torch
import torch.nn as nn

torch.set_float32_matmul_precision("high")

from .data import N_JOINTS, N_LEGS, build_group_cols


# ═══════════════════════════════════════════════════════════════════
# Stateful SNN
# ═══════════════════════════════════════════════════════════════════

def spike_fn(x, slope=25.0):
    """Heaviside forward, fast-sigmoid surrogate backward, from plain tensor ops (a custom
    autograd.Function is a Dynamo graph-break source and an opaque box to the fuser).

        surr = x / (slope*|x| + 1)      d(surr)/dx = 1 / (slope*|x| + 1)^2
        return hard + (surr - surr.detach())          hard = (x > 0)

    The parentheses matter: `surr - surr.detach()` is exactly 0.0 in IEEE-754, so the
    forward is bit-exactly `hard`; unparenthesized, float non-associativity gives
    1.0 +/- 1 ulp. Backward: only the tracked `surr` term carries gradient.
    """
    surr = x / (slope * x.abs() + 1.0)
    hard = (x > 0).to(x.dtype)
    return hard + (surr - surr.detach())


# Weight init scales (internal: neither needs user input). _W_IN_INIT: CPG->timing weights;
# calibrate_gains rescales them, so only the shape and the sign of the mean matter.
# _W1_INIT: layer 1; uncalibrated and learnable, so a starting point (inert under
# --sub_ln l1/both, where LN is scale-invariant).
_W_IN_INIT = 0.5
_W1_INIT   = 0.5

# Drive-LIF membrane tau for cpg_lif, in burst widths. Fixed because both ends are pinned:
# it must forget a full cycle of silence (a unit sees one burst per cycle) yet not within
# a burst (so a low gain can integrate several spikes, giving sub-burst placement). At 2
# burst widths (burst_width = period / n_cpg) the residual after a cycle is exp(-n_cpg/2),
# 1.2e-4 at n_cpg 18, independent of burst width, while retention across one ISI is ~0.9;
# the gain to fire on spike 1..5 of a 5-spike burst is 1.00/0.52/0.37/0.29/0.24.
_TAU_DRIVE_BURSTS = 2.0

# Initial s_gate under cpg_lif: mean 1.1, std 0.01. The spread breaks an exact tie: all-ones
# makes the drive exactly uniform, PR's stationary maximum, so dPR/ds is exactly 0 and the
# penalty does nothing (under cpg, u's row-norm spread breaks it by accident). Mean 1.1,
# not 1.0: spike_fn uses a strict >, so s=1.0 fires on spikes 2 and 4 only (2/5) while
# s=1.01 fires on all 5, and 1.1 starts every unit cleanly "all on". Small spread because
# PR is rich-get-richer: in a toy, a wide N(0.8, 0.4) init let the init's strongest
# channel win 90% of the time against 52% for a tight one.
_S_GATE_INIT_MEAN   = 1.1
_S_GATE_INIT_SPREAD = 0.01

# Smallest gap kept between a voltage-mode bias and `thresh` (see clamp_bias_voltage).
_V_EPS = 1e-6

def init_beta_logit(shape, tau_min, tau_max, generator=None):
    """Log-uniform membrane time constants -> beta = exp(-1/tau), stored as a logit so
    training keeps beta in (0,1). The most important choice in the model: with no
    phase channel, only units that decay slowly enough to still carry the last burst
    can know where in the cycle they are; a homogeneous beta=0.9 cannot.
    """
    u   = torch.rand(shape, generator=generator)
    tau = torch.exp(math.log(tau_min) + u * (math.log(tau_max) - math.log(tau_min)))
    beta = torch.exp(-1.0 / tau).clamp(1e-4, 1 - 1e-6)
    return torch.log(beta / (1.0 - beta))


class DenseSNN(nn.Module):
    """Input: n_neurons binary CPG spikes per step (nothing else). Gait: integer index,
    used only for FiLM. Output: n_joints joint angles per step.

    Fully connected: every CPG spike reaches every hidden unit, both hidden layers and
    the readout are dense, with no grouping or per-gait input routing. Purely
    feedforward within a step; all memory is in the three leaky membranes, whose
    learnable per-unit taus (log-uniform over [tau_min, tau_max]) are the only way to
    hold position within a gait cycle, so tau_max must comfortably exceed one period.

    FiLM: `film1`/`film2` are nn.Embedding lookup tables mapping a gait index to a
    (gamma, beta) per hidden unit, applied AFTER LayerNorm (a gait flag concatenated
    onto the input would have its uniform part removed and its scale normaliized away
    by LN). The tables are over-allocated to `max_gaits` rows (unused rows are identity
    modulation and get no gradient), so parameter shapes do not depend on the gait
    count.

    State: mem1, mem2 (B, H); memo (B, Ho), which equals H unless --readout_hidden
    narrows it.

    readout_hidden, tau_readout_max, bias_mode, hidden_reset, sub_ln and sub_film mean
    what they mean in TimingGroupedSNN, so this ablation differs from the grouped model
    only in the factorization. Defaults are the values this class used before the
    options existed, so old checkpoints reconstruct as what they ran.
    """

    # Consumed by export_onnx / config so neither has to branch on isinstance.
    arch            = "dense"
    state_names_in  = ("mem1_in",  "mem2_in",  "memo_in")
    state_names_out = ("mem1_out", "mem2_out", "memo_out")

    def __init__(self, hidden=128, n_gaits=4, max_gaits=16,
                 tau_min=2.0, tau_max=256.0,
                 slope=25.0, thresh=1.0, n_neurons=4, n_joints=N_JOINTS,
                 readout_hidden=None, tau_readout_max=40.0,
                 bias_mode="current", hidden_reset="subtract",
                 sub_ln="both", sub_film="both", film_mode="gamma_only"):
        super().__init__()
        if n_gaits > max_gaits:
            raise ValueError(
                f"n_gaits ({n_gaits}) > max_gaits ({max_gaits}); raise "
                f"--max_gaits. Note that changing max_gaits changes the FiLM "
                f"parameter shape and so invalidates old checkpoints.")
        for nm, val, ok in (("bias_mode", bias_mode,
                             ("current", "voltage", "none")),
                            ("hidden_reset", hidden_reset,
                             ("subtract", "zero")),
                            ("sub_ln", sub_ln, ("none", "l1", "l2", "both")),
                            ("sub_film", sub_film,
                             ("none", "l1", "l2", "both"))):
            if val not in ok:
                raise ValueError(f"{nm} must be one of {ok}, got {val!r}")
        self.H         = hidden
        # None = as wide as the hidden layers (what this class did before the option existed).
        self.Ho        = int(readout_hidden) if readout_hidden else hidden
        self.bias_mode    = bias_mode
        self.hidden_reset = hidden_reset
        self.sub_ln       = sub_ln
        self.sub_film     = sub_film
        self.film_mode    = film_mode
        self.n_gaits   = n_gaits
        self.max_gaits = max_gaits
        self.slope     = slope
        self.thresh    = thresh
        # Stored so export_onnx can size the dummy input.
        self.n_neurons = n_neurons
        self.n_joints  = n_joints

        # ── layer 1: all CPG spikes -> all hidden units ─────────────
        # Scale 0.8: one CPG neuron fires per step, so cur1 is a single row of w_in; LayerNorm
        # follows, so this mostly sets gradient scale.
        self.w_in   = nn.Parameter(torch.randn(n_neurons, hidden) * 0.8)
        self.ln1    = nn.LayerNorm(hidden)

        # ── layer 2 ───────────────────────────────────────────────
        self.w2     = nn.Parameter(torch.randn(hidden, hidden) / math.sqrt(hidden))
        self.ln2    = nn.LayerNorm(hidden)

        # ── readout: non-spiking leaky membrane, then joint angles ─
        Ho = self.Ho
        self.w_read = nn.Parameter(torch.randn(hidden, Ho) / math.sqrt(hidden))
        self.w_out  = nn.Parameter(torch.randn(Ho, n_joints) / math.sqrt(Ho))
        self.b_out  = nn.Parameter(torch.zeros(n_joints))

        # ── bias: additive current, or an offset to the threshold ─
        # b_read has no voltage equivalent (memo is analog, never spikes), so it exists only in
        # "current" mode; b_out carries the output offset.
        if bias_mode == "current":
            self.b1     = nn.Parameter(torch.zeros(hidden))
            self.b2     = nn.Parameter(torch.zeros(hidden))
            self.b_read = nn.Parameter(torch.zeros(Ho))
        elif bias_mode == "voltage":
            self.v1 = nn.Parameter(torch.zeros(hidden))
            self.v2 = nn.Parameter(torch.zeros(hidden))

        # ── heterogeneous, learnable time constants ───────────────
        self.beta1_logit = nn.Parameter(init_beta_logit((hidden,), tau_min, tau_max))
        self.beta2_logit = nn.Parameter(init_beta_logit((hidden,), tau_min, tau_max))
        self.betao_logit = nn.Parameter(
            init_beta_logit((self.Ho,), 2.0, float(tau_readout_max)))

        # ── FiLM: per-gait scale/shift, over-allocated ────────────
        if sub_film in ("l1", "both"):
            self.film1 = nn.Embedding(max_gaits, 2 * hidden)
        if sub_film in ("l2", "both"):
            self.film2 = nn.Embedding(max_gaits, 2 * hidden)
        for e in (getattr(self, "film1", None), getattr(self, "film2", None)):
            if e is not None:
                nn.init.zeros_(e.weight)
                e.weight.data[:, :hidden] = 1.0   # gamma := 1, beta := 0

    # ---------------------------------------------------------------
    def init_state(self, batch, device, dtype=torch.float32):
        """(mem1, mem2, memo). memo is Ho wide, which is H unless narrowed."""
        z = lambda w: torch.zeros(batch, w, device=device, dtype=dtype)
        return (z(self.H), z(self.H), z(self.Ho))

    def step(self, x, gait, state):
        """
        x     : (B, n_neurons) float — CPG spikes this timestep
        gait  : (B,) int64
        state : 3-tuple (mem1, mem2, memo); mem1/mem2 are (B, H) and
                memo is (B, Ho)

        Feedforward within a timestep: spk1/spk2 are local, consumed by the
        next layer in the same step and never carried across steps.
        `addmm` folds each bias into its matmul, one kernel per layer.
        """
        mem1, mem2, memo = state
        cur_bias = self.bias_mode == "current"

        # ---- layer 1 -------------------------------------------------
        cur1 = (torch.addmm(self.b1, x, self.w_in) if cur_bias
                else x @ self.w_in)
        if self.sub_ln in ("l1", "both"):
            cur1 = self.ln1(cur1)
        if self.sub_film in ("l1", "both"):
            v1 = self.film1(gait)                          # (B, 2H)
            # See TimingGroupedSNN's film_mode note: gamma is multiplicative
            # and vanishes with the injection, beta is tonic drive.
            cur1 = cur1 * v1[:, :self.H]
            if self.film_mode == "gamma_beta":
                cur1 = cur1 + v1[:, self.H:]
        beta1 = torch.sigmoid(self.beta1_logit)
        mem1  = beta1 * mem1 + cur1
        # Comparison threshold only: the reset below subtracts the fixed
        # self.thresh, matching TimingGroupedSNN. Resetting by the offset
        # threshold instead lets a unit that learns a low threshold fire
        # easily AND reset by almost nothing, which runs away.
        th1  = self.thresh - self.v1 if self.bias_mode == "voltage" else self.thresh
        spk1 = spike_fn(mem1 - th1, self.slope)
        mem1 = (mem1 * (1.0 - spk1) if self.hidden_reset == "zero"
                else mem1 - self.thresh * spk1)

        # ---- layer 2 -------------------------------------------------
        cur2 = (torch.addmm(self.b2, spk1, self.w2) if cur_bias
                else spk1 @ self.w2)
        if self.sub_ln in ("l2", "both"):
            cur2 = self.ln2(cur2)
        if self.sub_film in ("l2", "both"):
            v2 = self.film2(gait)
            cur2 = cur2 * v2[:, :self.H]
            if self.film_mode == "gamma_beta":
                cur2 = cur2 + v2[:, self.H:]
        beta2 = torch.sigmoid(self.beta2_logit)
        mem2  = beta2 * mem2 + cur2
        th2  = self.thresh - self.v2 if self.bias_mode == "voltage" else self.thresh
        spk2 = spike_fn(mem2 - th2, self.slope)
        mem2 = (mem2 * (1.0 - spk2) if self.hidden_reset == "zero"
                else mem2 - self.thresh * spk2)

        # ---- analog readout -----------------------------------------
        curo  = (torch.addmm(self.b_read, spk2, self.w_read) if cur_bias
                 else spk2 @ self.w_read)
        betao = torch.sigmoid(self.betao_logit)
        memo  = betao * memo + curo

        y = torch.addmm(self.b_out, memo, self.w_out)       # (B, n_joints)
        # aux is None here (no timing layer to report); the 3-tuple keeps
        # step()'s contract identical across both architectures so callers
        # never branch on type.
        return y, (mem1, mem2, memo), None

    def forward(self, x_seq, gait_seq, state=None, return_aux=False):
        """
        x_seq    : (L, B, n_neurons)
        gait_seq : (L, B)

        Returns (y_seq, state), or (y_seq, state, None) when return_aux=True
        -- this arch has no timing layer, so there are no spikes to report,
        but the signature matches TimingGroupedSNN so run_training does not
        have to know which model it has.
        """
        L, B = x_seq.shape[0], x_seq.shape[1]
        if state is None:
            state = self.init_state(B, x_seq.device, x_seq.dtype)
        ys = []
        for t in range(L):
            y, state, _ = self.step(x_seq[t], gait_seq[t], state)
            ys.append(y)
        if return_aux:
            return torch.stack(ys), state, None
        return torch.stack(ys), state


class TimingGroupedSNN(nn.Module):
    """CPG spikes -> TIMING layer -> G disconnected sub-networks.

        x          (B, n_neurons)  CPG spikes, at most one fires per step
        timing     (B, n_timing)   LIF; per-gait weights from the CPG spikes
        sub-net g  (B, Hg) x2 + memo, driven by ITS OWN K timing units
        y          (B, n_joints)   group g writes only its own columns

    G and n_timing are independent: `group_cols[g]` is the output columns
    sub-network g owns, `timing_map[g]` the K timing units feeding it (from
    --decoder_shape / --timing_shape via build_partition / build_timing_map).
    Sub-networks never interact: a timing unit shared by several is a shared
    INPUT, not a shared parameter, so layers 2+ and the readout stay block diagonal.

    Per-gait routing: `w_in_gait` is Embedding(max_gaits, n_neurons * n_timing) and
    `b_t` a per-gait bias, so each gait has its own free routing matrix. A shared
    router with a per-gait FiLM gate was tried and reverted (the learned timing
    phases came out identical across gaits; see git history). Routing is not
    constrained to a permutation: a tripod needs three legs on one phase.

    bias_mode: "current" = additive bias current; "voltage" (default) = per-unit
    offset to the spiking threshold, with no bias current anywhere in the spiking
    path (what makes natural gating possible); "none". The reset still subtracts
    the fixed `thresh`, not the offset threshold (resetting by the offset runs
    away). b_read exists only in "current" mode (memo is analog and never spikes).

    No LayerNorm on the timing layer: over a handful of units it removes the
    common-mode "some CPG neuron fired" signal and turns the bias into tonic drive.
    `calibrate_gains` sets the firing regime instead.

    NATURAL GATING (the default setup): gate_mode="none", bias_mode="voltage",
    film_mode="gamma_only" and hidden_reset="zero" leave no way for a hidden unit
    to fire on a timing-silent step (no bias current, no FiLM beta, no residual
    after a subtractive reset). --bias_mode current, --film_mode gamma_beta and
    --hidden_reset subtract each reopen one leak, for A/B; check_bias_voltage.py
    reports which units can fire unprompted. With a leak open, the task loss is
    partly insensitive to WHEN the timing spikes land.

    Inside the sub-networks LN (`sub_ln`) uses elementwise_affine=False so no
    parameter is tied across groups; FiLM supplies the per-group per-gait affine.

    State (9 tensors), in this order:
        mem_timing (B, n_timing)  timing layer; unused under cpg / cpg_lif
        since_upd  (B, G)         vestigial; kept only so the ONNX signature is fixed
        mem1, mem2 (B, G, Hg)
        memo       (B, G, H_o)    narrower than Hg; see the readout note in __init__
        syn1, syn2 (B, G, Hg)     second-order synaptic currents (--synaptic)
        syno       (B, G, H_o)    zero when synaptic="none"
        mem_drive  (B, G, n_cpg)  drive LIF membrane (cpg_lif only), else zero
    """

    arch = "timing_grouped"
    # syn1/syn2/syno are the second-order synaptic currents. Allocated
    # unconditionally, like the alpha logits and like the vestigial
    # since_upd, so neither the state arity nor the exported ONNX signature
    # depends on --synaptic. They stay exactly zero when synaptic="none".
    # mem_drive is the drive LIF's membrane, (B, G, C). Allocated
    # unconditionally like syn*/since_upd so neither the state arity nor the
    # exported ONNX signature depends on --drive_source; it stays exactly
    # zero except under cpg_lif.
    state_names_in  = ("mem_timing_in", "since_upd_in",
                       "mem1_in",  "mem2_in",  "memo_in",
                       "syn1_in",  "syn2_in",  "syno_in",
                       "mem_drive_in")
    state_names_out = ("mem_timing_out", "since_upd_out",
                       "mem1_out", "mem2_out", "memo_out",
                       "syn1_out", "syn2_out", "syno_out",
                       "mem_drive_out")

    def __init__(self, hidden_per_group=128, n_gaits=4, max_gaits=16,
                 n_neurons=4, n_timing=N_LEGS, group_cols=None,
                 timing_map=None,
                 n_joints=N_JOINTS, readout_hidden=32,
                 tau_min=2.0, tau_max=256.0,
                 tau_timing_min=2.0, tau_timing_max=64.0,
                 tau_readout_max=40.0,
                 sub_ln="l2", sub_film="both", film_mode="gamma_only",
                 timing_reset="zero", hidden_reset="subtract", gate_mode="none",
                 bias_mode="voltage", readout_gait_bias=True,
                 synaptic="none", tau_syn_min=2.0, tau_syn_max=20.0,
                 drive_source="timing", tau_drive=None,
                 slope=25.0, timing_slope=None, thresh=1.0):
        super().__init__()
        if n_gaits > max_gaits:
            raise ValueError(
                f"n_gaits ({n_gaits}) > max_gaits ({max_gaits}); raise "
                f"--max_gaits. Note that changing max_gaits changes the FiLM "
                f"parameter shapes and so invalidates old checkpoints.")
        if sub_ln not in ("none", "l1", "l2", "both"):
            raise ValueError(f"sub_ln must be none|l1|l2|both, got {sub_ln!r}")
        if sub_film not in ("none", "l1", "l2", "both"):
            raise ValueError(f"sub_film must be none|l1|l2|both, got {sub_film!r}")
        if timing_reset not in ("zero", "subtract"):
            raise ValueError(f"timing_reset must be zero|subtract, got "
                             f"{timing_reset!r}")
        if film_mode not in ("gamma_only", "gamma_beta"):
            raise ValueError(f"film_mode must be gamma_only|gamma_beta, got "
                             f"{film_mode!r}")
        if synaptic not in ("none", "hidden", "all"):
            raise ValueError(f"synaptic must be none|hidden|all, got "
                             f"{synaptic!r}")
        if drive_source not in ("timing", "cpg", "cpg_lif"):
            raise ValueError(f"drive_source must be timing|cpg|cpg_lif, got "
                             f"{drive_source!r}")
        if drive_source in ("cpg", "cpg_lif") and gate_mode != "none":
            raise ValueError(
                "drive_source cpg / cpg_lif requires gate_mode='none' (there is no timing "
                "unit to gate on).")

        group_cols = (build_group_cols(n_timing, n_joints=n_joints)
                      if group_cols is None else
                      [list(g) for g in group_cols])

        G = len(group_cols)
        T = int(n_timing)
        # timing_map[g] = timing units feeding sub-network g; None means one per
        # sub-network, in order (what configs predating the arg were).
        if timing_map is None:
            if T != G:
                raise ValueError(
                    f"group_cols has {G} groups but n_timing={T} and no timing_map was given; "
                    f"pass timing_map (see build_timing_map) for a fan-in other than 1.")
            timing_map = [[g] for g in range(G)]
        else:
            timing_map = [list(m) for m in timing_map]
            if len(timing_map) != G:
                raise ValueError(
                    f"timing_map has {len(timing_map)} entries but there are "
                    f"{G} sub-networks.")
        K = len(timing_map[0])
        if any(len(m) != K for m in timing_map):
            raise ValueError(
                f"timing_map fan-in must be uniform; got sizes "
                f"{sorted({len(m) for m in timing_map})}.")
        bad = [i for m in timing_map for i in m if not 0 <= i < T]
        if bad:
            raise ValueError(f"timing_map indexes units {sorted(set(bad))} "
                             f"outside 0..{T - 1}.")

        Hg  = int(hidden_per_group)
        Ho  = int(readout_hidden)
        C   = len(group_cols[0])            # output columns per group

        self.G          = G
        self.Hg         = Hg
        self.Ho         = Ho
        self.C          = C
        self.K          = K           # timing units feeding each sub-network
        # Under cpg / cpg_lif the drive channels are the CPG's own, so n_timing is
        # n_neurons (timing_only returns the CPG train there; diagnostics key off this).
        self.n_timing   = T if drive_source == "timing" else int(n_neurons)
        self.timing_map_list = [list(m) for m in timing_map]
        self.register_buffer(
            "timing_map",
            torch.tensor(timing_map, dtype=torch.long).view(G, K),
            persistent=False)
        self.n_neurons  = int(n_neurons)
        self.n_joints   = int(n_joints)
        self.n_gaits    = int(n_gaits)
        self.max_gaits  = int(max_gaits)
        self.slope      = slope
        # Gentler surrogate on the timing layer, where a dead unit is catastrophic.
        self.timing_slope = float(slope if timing_slope is None
                                  else timing_slope)
        self.thresh     = thresh
        self.sub_ln     = sub_ln
        self.sub_film   = sub_film
        self.film_mode  = film_mode
        self.synaptic   = synaptic
        self.drive_source = drive_source
        self.uses_timing  = (drive_source == "timing")
        # cpg_lif shares every parameter with cpg and adds the drive LIF.
        self.drive_lif    = (drive_source == "cpg_lif")
        self.timing_reset = timing_reset
        self.hidden_reset = hidden_reset
        if gate_mode not in ("none", "decay"):
            raise ValueError(f"gate_mode must be none|decay, got "
                             f"{gate_mode!r}")
        self.gate_mode = gate_mode
        if bias_mode not in ("current", "voltage", "none"):
            raise ValueError(f"bias_mode must be current|voltage|none, got "
                             f"{bias_mode!r}")
        self.bias_mode = bias_mode
        self.group_cols = group_cols
        self.H          = Hg          # alias for generic callers

        # ── output column routing ─────────────────────────────────
        flat = [c for grp in group_cols for c in grp]
        inv  = np.empty(n_joints, dtype=np.int64)
        for pos, col in enumerate(flat):
            inv[col] = pos
        self.register_buffer("out_perm", torch.from_numpy(inv), persistent=False)

        # ── timing: per-gait CPG->timing weights ─────────────────
        # Each gait has its own free (n_neurons, n_timing) matrix and bias. No FiLM here:
        # with a free per-gait W, gamma is absorbable into W (the per-gait bias is not,
        # so it comes from the embedding). Init scale is arbitrary (calibrate_gains
        # rescales it) but the mean must be positive: calibration cannot rescue a
        # net-negative unit.
        # Timing layer skipped under cpg / cpg_lif, which also makes calibrate_gains /
        # reinit_timing_units (keyed on hasattr w_in_gait) no-ops.
        if self.uses_timing:
            self.w_in_gait = nn.Embedding(max_gaits, self.n_neurons * T)
            nn.init.normal_(self.w_in_gait.weight,
                            mean=_W_IN_INIT, std=_W_IN_INIT)

            # Per-gait excitability (see BIAS MODES). "current" = tonic drive, the only way a
            # unit fires BETWEEN bursts under timing_reset=zero; "voltage" = threshold offset.
        if self.uses_timing:
            if bias_mode == "current":
                self.b_t = nn.Embedding(max_gaits, T)
                nn.init.zeros_(self.b_t.weight)
            elif bias_mode == "voltage":
                self.v_t = nn.Embedding(max_gaits, T)
                nn.init.zeros_(self.v_t.weight)
            self.beta_t_logit = nn.Parameter(
                init_beta_logit((T,), tau_timing_min, tau_timing_max))

        # ── sub-network layer 1: K binary spikes -> Hg units ──────
        # Fixed init, not calibrated (a few dead units out of Hg are noise). Shape
        # (G, K, Hg), contracted over K in `step`; 1/sqrt(K) keeps the injection
        # variance independent of fan-in (exactly 1.0 at K=1). Created only when
        # uses_timing; the cpg paths have s_gate/u instead.
        if self.uses_timing:
            self.w1 = nn.Parameter(
                torch.randn(G, K, Hg) * (_W1_INIT / math.sqrt(K)))
        else:
        # ── drive_source cpg / cpg_lif: skip the timing layer ─────
        # Sub-networks are driven straight off the CPG channels, whose phases are fixed
        # and evenly tiled. The per-gait selection w_in_gait provided is factorized:
        #   s_gate (max_gaits, G*n_ch)  gain of channel c on sub-network g, per gait,
        #       applied BEFORE the sum over channels, so 0 removes that channel from that
        #       (gait, sub-network) exactly.
        #   u (G, n_ch, Hg)  the layer-1 weights (w1's counterpart), gait-SHARED: a gait
        #       can scale a channel's hidden-space direction but not rotate it (rank 1
        #       over gait; ~7.5x fewer parameters than a full per-gait w1).
        #   cur1[b,g,h] = sum_c x[b,c] * s_gate[gait,g,c] * u[g,c,h]
        # n_ch, not C: C is the OUTPUT-column count and sizes w_out / b_out below
        # (rebinding it here once routed every joint through sub-network 0).
            n_ch = self.n_neurons
            self.s_gate = nn.Embedding(max_gaits, G * n_ch)
            nn.init.ones_(self.s_gate.weight)     # every channel on
            if self.drive_lif:
                # See _S_GATE_INIT_*. cpg_lif only: under cpg, u's row norms already break the tie.
                with torch.no_grad():
                    self.s_gate.weight.fill_(_S_GATE_INIT_MEAN)
                    self.s_gate.weight.add_(
                        torch.randn_like(self.s_gate.weight)
                        * _S_GATE_INIT_SPREAD)
                    self.s_gate.weight.clamp_(min=1e-3)

            # Dense at _W1_INIT with no 1/sqrt(C) fan-in factor: CPG channels never co-fire, so
            # a hidden unit receives ONE weight per spiking step, not a sum. Dense (not
            # sparse) because a unit needs some positive weight to ever fire (fan-in 2
            # leaves ~25% dead) and argmax_c u[g,c,h] already tiles phase across units.
            self.u = nn.Parameter(torch.randn(G, n_ch, Hg) * _W1_INIT)

            if self.drive_lif:
                # ── drive_source cpg_lif ─────────────────────────────────
                # A LIF between the gain and u: one unit per (g, c), each seeing exactly one
                # channel, so it can thin or delay that channel's burst but never move it.
                # Why: under cpg, s_gate and u are multiplicatively redundant (halve s, double u
                # = identical forward), so a penalty on s is meaningless. The threshold bounds
                # that escape: dropped spikes lose information u cannot restore (timing within
                # the burst, spike count), and below s ~0.24 a channel is silent. Spike count is
                # a staircase in s, so the penalty acts on |s_gate| instead (smooth).
                # FIXED threshold: a learnable one reopens the escape.
                # Zero reset: makes a sub-network's C units mutually exclusive (one channel each,
                # CPG channels never coincide), so they collapse to one effective timing neuron.
                # FIXED tau (buffer): both ends are pinned by the burst structure.
                if tau_drive is None:
                    raise ValueError(
                        "drive_source='cpg_lif' needs tau_drive (main() passes "
                        "_TAU_DRIVE_BURSTS * period / n_cpg_neurons; the constructor does not know "
                        "the period).")
                self.tau_drive = float(tau_drive)
                self.register_buffer(
                    "drive_decay",
                    torch.tensor(math.exp(-1.0 / max(self.tau_drive, 1e-6)),
                                 dtype=torch.float32),
                    persistent=False)

        # ── sub-network layer 2: block diagonal (G, Hg, Hg) ───────
        self.w2 = nn.Parameter(torch.randn(G, Hg, Hg) / math.sqrt(Hg))

        # ── block-diagonal analog readout ─────────────────────────
        # Ho, not Hg: memo is a bank of low-pass filters (one tau per unit), so its width
        # is the temporal basis for the output waveform, not the output width. Ho == C
        # would leave only C distinct (projection, tau) pairs per group.
        # C must be the OUTPUT-column count (asserted: a wrong C still trains, with 17
        # of 18 sub-networks disconnected).
        assert C == len(group_cols[0]), (
            f"C was rebound: expected len(group_cols[0])="
            f"{len(group_cols[0])}, got {C}. w_out/b_out would be sized "
            f"wrong and out_perm would select only sub-network 0.")
        self.w_read = nn.Parameter(torch.randn(G, Hg, Ho) / math.sqrt(Hg))
        self.w_out  = nn.Parameter(torch.randn(G, Ho, C) / math.sqrt(Ho))
        self.b_out  = nn.Parameter(torch.zeros(G, C))

        # b_read has no voltage equivalent (memo is analog and never spikes), so it exists
        # only in "current" mode; b_out carries the output offset.
        if bias_mode == "current":
            self.b1     = nn.Parameter(torch.zeros(G, Hg))
            self.b2     = nn.Parameter(torch.zeros(G, Hg))
            self.b_read = nn.Parameter(torch.zeros(G, Ho))
        elif bias_mode == "voltage":
            self.v1 = nn.Parameter(torch.zeros(G, Hg))
            self.v2 = nn.Parameter(torch.zeros(G, Hg))

        self.ln1 = nn.LayerNorm(Hg, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(Hg, elementwise_affine=False)

        self.beta1_logit = nn.Parameter(init_beta_logit((G, Hg), tau_min, tau_max))
        self.beta2_logit = nn.Parameter(init_beta_logit((G, Hg), tau_min, tau_max))
        self.betao_logit = nn.Parameter(
            init_beta_logit((G, Ho), 2.0, tau_readout_max))

        self.film1 = nn.Embedding(max_gaits, 2 * G * Hg)
        self.film2 = nn.Embedding(max_gaits, 2 * G * Hg)
        for e in (self.film1, self.film2):
            nn.init.zeros_(e.weight)
            e.weight.data[:, :G * Hg] = 1.0

        # ── per-gait readout bias current ────────────────────────
        # A per-gait DC current into `memo` (not added to y, so it fades in over the
        # readout tau at a gait switch instead of stepping). Needed because nothing else
        # in the sub-networks has a gait-dependent constant under film_mode=gamma_only,
        # so gaits with different MEAN angles would have to express it through spike
        # rate. memo is analog and never spikes, so this adds no tonic drive to a
        # spiking layer. Created only when enabled.
        self.readout_gait_bias = bool(readout_gait_bias)
        if self.readout_gait_bias:
            self.b_read_gait = nn.Embedding(max_gaits, G * Ho)
            nn.init.zeros_(self.b_read_gait.weight)

        # ── second-order synaptic filter (snnTorch Synaptic / Loihi CUBA) ──
        #     syn = alpha * syn + injection;   mem = beta * mem + syn
        # (mem integrates the NEW syn, so the membrane still moves on the spike step,
        # just by less.) One injection is spread over many steps: a smaller output step
        # per spike, and hidden membranes stay near threshold for a window, so the
        # surrogate gradient stays alive on silent steps. It decays to zero in true
        # silence, so spike placement still matters. alpha logits are always allocated
        # (no gradient when synaptic="none").
        self.alpha1_logit = nn.Parameter(
            init_beta_logit((G, Hg), tau_syn_min, tau_syn_max))
        self.alpha2_logit = nn.Parameter(
            init_beta_logit((G, Hg), tau_syn_min, tau_syn_max))
        self.alphao_logit = nn.Parameter(
            init_beta_logit((G, Ho), tau_syn_min, tau_syn_max))

    def drive_strength(self):
        """(n_gaits, G, C) drive of CPG channel c on sub-network g, per gait.

        cpg:      |s_gate| * ||u[g,c,:]||_1. s_gate and u enter the forward pass only as
                  a product, so neither alone says how much current a channel delivers.
        cpg_lif:  |s_gate|. A fixed-threshold LIF sits between s_gate and u, so u gets
                  binary spikes and the penaliized quantity is the gain itself.

        Shared by `drive_penalty` and the reporters, so the penaliized and reported
        numbers are the same by construction. Only the first n_gaits rows are used.
        """
        S = self.s_gate.weight[:self.n_gaits].abs().view(
            self.n_gaits, self.G, self.n_neurons)               # (n_gaits,G,C)
        if self.drive_lif:
            return S
        unorm = self.u.abs().sum(dim=2)                         # (G, C)
        return S * unorm.unsqueeze(0)

    @torch.no_grad()
    def drive_spikes_only(self, x_seq, gait_seq, mem_d=None):
        """Replay just the drive LIF layer: (L, B, G, C) binary spikes, or None outside
        cpg_lif. The cpg_lif counterpart of `timing_only`; "channel c delivers nothing to
        sub-network g" is the exact statement "unit (g,c) emitted zero spikes".
        """
        if not self.drive_lif:
            return None
        B = x_seq.shape[1]
        dev, dt = x_seq.device, x_seq.dtype
        if mem_d is None:
            mem_d = torch.zeros(B, self.G, self.n_neurons,
                                device=dev, dtype=dt)
        dec = self.drive_decay
        out = []
        for t in range(x_seq.shape[0]):
            s_g = self.s_gate(gait_seq[t]).view(-1, self.G, self.n_neurons)
            mem_d = dec * mem_d + x_seq[t].unsqueeze(1) * s_g
            spk = (mem_d > self.thresh).to(dt)
            mem_d = mem_d * (1.0 - spk)
            out.append(spk)
        return torch.stack(out)

    def drive_penalty(self, kind="pr"):
        """Sparsity pressure on the drive (cpg / cpg_lif; see drive_strength for what the
        drive is), per (gait, sub-network) over that row's C channels.

        kind="pr" (default): participation ratio (sum d)^2 / sum d^2, the effective
            number of channels used (1 = one channel, C = all equal); the same quantity
            drive_report prints as fanin. Scale-invariant, so shrinking every drive and
            compensating elsewhere cannot lower it. Rich-get-richer: it amplifies the
            leading channel, and exactly-uniform is its (unstable) maximum, broken by u's
            row-norm spread under cpg and _S_GATE_INIT_SPREAD under cpg_lif. A weight
            that dominates the task gradient lets init pick the winner, so start small
            and rely on --lambda_hold / --lambda_warmup.
        kind="l1": mean of the drive. Under cpg it FAILED (the network shrank every drive
            together and compensated via u and the hidden thresholds: max channel share
            0.08 at fanin ~15.8), and its gradient ||u[g,c,:]||_1 barely varies across
            channels. Under cpg_lif the drive is |s_gate| and it is UNTESTED; the escape
            is bounded: s_gate can fall from 1.1 to ~0.25 (one late-burst spike) while u,
            FiLM gamma and v1 make up the charge, but dropped spikes lose timing within
            the burst and spike count.

        Returns a 0-d tensor; zero when there is no s_gate.
        """
        if self.uses_timing:
            return torch.zeros((), device=self.w2.device, dtype=self.w2.dtype)
        d = self.drive_strength()                       # (n_gaits, G, C)
        if kind == "l1":
            return d.mean()
        if kind != "pr":
            raise ValueError(f"drive penalty kind must be pr|l1, got {kind!r}")
        s1 = d.sum(dim=2)
        s2 = d.pow(2).sum(dim=2).clamp(min=1e-12)
        # Mean over (gait, sub-network), divided by C, so the weight needs no rescaling
        # when n_gaits, G or n_cpg_neurons change.
        return (s1.pow(2) / s2).mean() / float(self.n_neurons)

    @torch.no_grad()
    def clamp_bias_voltage(self):
        """Project v_t / v1 / v2 into [-thresh, thresh - _V_EPS].

        The upper bound matters: a voltage-mode unit compares against `thresh - v`, so
        v >= thresh gives a non-positive threshold and the unit fires every step with no
        way back. The lower bound only caps the threshold at 2*thresh.

        Applied to `.data` after the optimiizer step, not in `step()`, where a clamp
        would give out-of-range parameters exactly zero gradient and trap them. Not
        called from load_run: a checkpoint must reconstruct as it was
        (check_bias_voltage.py finds out-of-range biases).
        """
        if self.bias_mode != "voltage":
            return
        lo, hi = -self.thresh, self.thresh - _V_EPS
        # v_t exists only when there IS a timing layer (drive_source="timing").
        v_t = getattr(self, "v_t", None)
        tensors = [self.v1, self.v2]
        if v_t is not None:
            tensors.append(v_t.weight)
        for p in tensors:
            p.data.clamp_(lo, hi)

    def init_state(self, batch, device, dtype=torch.float32):
        z = lambda w: torch.zeros(batch, self.G, w, device=device, dtype=dtype)
        mem_t = torch.zeros(batch, self.n_timing, device=device, dtype=dtype)
        # since_upd: vestigial (the update counter of the removed gate_mode="freeze"); kept
        # only so the ONNX signature keeps its arity and names.
        since = torch.zeros(batch, self.G, device=device, dtype=dtype)
        # memo is Ho wide, not Hg; syn1/syn2/syno mirror mem1/mem2/memo.
        return (mem_t, since, z(self.Hg), z(self.Hg), z(self.Ho),
                z(self.Hg), z(self.Hg), z(self.Ho), z(self.n_neurons))

    def _timing(self, x, gait, mem_t):
        """One timestep of the timing layer: x (B, n_neurons) -> spk_t (B, T). `step` maps
        these onto sub-networks through `timing_map`. Factored out so calibration and
        diagnostics can run it without the sub-networks or the compiled `step`.

        timing_reset="zero" matches the CPG (LIFGeneralArray does v[spike]=0). A
        subtractive reset leaves mem - thresh behind, which can re-fire on the next
        silent step: ISIs of 2,1,1,2,... and ~50% more spikes per burst than the CPG.
        """
        W   = self.w_in_gait(gait).view(-1, self.n_neurons, self.n_timing)
        cur = torch.bmm(x.unsqueeze(1), W).squeeze(1)
        if self.bias_mode == "current":
            cur = cur + self.b_t(gait)
        # Comparison threshold only; the reset still subtracts the fixed self.thresh
        # (resetting by the offset runs away).
        th = (self.thresh - self.v_t(gait) if self.bias_mode == "voltage"
              else self.thresh)
        mem_t = torch.sigmoid(self.beta_t_logit) * mem_t + cur
        spk_t = spike_fn(mem_t - th, self.timing_slope)
        if self.timing_reset == "zero":
            mem_t = mem_t * (1.0 - spk_t)
        else:
            mem_t = mem_t - self.thresh * spk_t
        return spk_t, mem_t

    def step(self, x, gait, state):
        """x : (B, n_neurons) CPG spikes this step;  gait : (B,) int64
        state : (mem_timing, since_upd, mem1, mem2, memo, syn1, syn2, syno, mem_drive)

        Returns (y, state, aux), aux = (spk_timing,): lets the spike penalty see the
        spikes without a second forward pass (the ONNX wrappers discard it).
        """
        (mem_t, since, mem1, mem2, memo,
         syn1, syn2, syno, mem_d) = state
        G, Hg = self.G, self.Hg

        if self.uses_timing:
            spk_t, mem_t = self._timing(x, gait, mem_t)      # (B, T)
            # ---- timing -> sub-network routing: timing_map is (G, K), gathering each
            # sub-network's K timing spikes ----
            spk_sel = spk_t[:, self.timing_map]              # (B, G, K)
        else:
            # drive_source cpg / cpg_lif: no timing layer to run. mem_t passes through (all-zero;
            # it only holds the state/ONNX signature fixed). spk_t is the CPG vector itself (an
            # input, no gradient), reported in aux for diagnostics. main() forces the spike
            # objective to none in these modes.
            spk_t   = x
            spk_sel = None

        # ---- event gate ----
        # gate_mode "none" (default) is NATURAL GATING (see the class docstring). "decay" gates
        # the input and sub-net spikes explicitly, kept for A/B. Not the default: it destroys
        # the magnitude-to-spike-count expansion of subtractive reset, and it changes the
        # timing layer's gradient. The gate is spk_t, so it multiplies spk_t in five extra
        # places and a path with k gate factors has gradient (1+k)*s^k: up to 6x on a spiking
        # step, but EXACTLY ZERO on a silent step, which removes the "fire HERE" signal. Dead
        # timing units are therefore MORE likely under it.
        gated = self.gate_mode != "none"
        # A sub-network updates when ANY of its K timing units fires: the OR over its selected
        # spikes (amax keeps the gate strictly 0/1; a no-op at K=1).
        gate = spk_sel.amax(dim=-1, keepdim=True) if gated else None
        dec1 = torch.sigmoid(self.beta1_logit)
        dec2 = torch.sigmoid(self.beta2_logit)
        deco = torch.sigmoid(self.betao_logit)

        # Injection -> membrane, first order (straight in) or second order (via a decaying
        # synaptic current; snnTorch ordering, mem integrates the NEW syn). Written once so
        # the three layers cannot drift; syn passes through unchanged (zero) in first order.
        def integrate(mem, syn, cur, dec, alpha_logit, second_order):
            if not second_order:
                return dec * mem + cur, syn
            syn = torch.sigmoid(alpha_logit) * syn + cur
            return dec * mem + syn, syn

        syn_hidden  = self.synaptic in ("hidden", "all")
        syn_readout = self.synaptic == "all"

        # ---- sub-net layer 1 ----
        # In "voltage" mode the bias is not in the injection, so it cannot inflate it. The
        # contraction over K is explicit in both gate modes: with K>1 the injection is
        # sum_k spk_k * w1[g,k], which is NOT gate * sum_k w1[g,k] unless all K units fire.
        b1 = self.b1 if self.bias_mode == "current" else 0.0
        if self.uses_timing:
            cur1 = torch.einsum("bgk,gkh->bgh", spk_sel, self.w1) + b1
        else:
            # Gate each CPG channel per (gait, sub-network) BEFORE the sum over channels, so
            # s_gate[gait,g,c]=0 removes exactly that channel from that sub-network and gait.
            s_g    = self.s_gate(gait).view(-1, G, self.n_neurons)  # (B,G,C)
            scaled = x.unsqueeze(1) * s_g                           # (B,G,C)
            if self.drive_lif:
                # One LIF per (g, c): fixed threshold, zero reset. Its binary OUTPUT is what reaches u,
                # which bounds how far a penalty on s_gate can be escaped (see __init__).
                mem_d = self.drive_decay * mem_d + scaled
                spk_d = spike_fn(mem_d - self.thresh, self.slope)
                mem_d = mem_d * (1.0 - spk_d)            # zero reset, always
                drive = spk_d
            else:
                drive = scaled
            cur1   = torch.einsum("bgc,gch->bgh", drive, self.u) + b1
        if self.sub_ln in ("l1", "both"):
            cur1 = self.ln1(cur1)
        if self.sub_film in ("l1", "both"):
            v1 = self.film1(gait).view(-1, 2, G, Hg)
            # gamma always; beta only in gamma_beta. Gamma is multiplicative, so it vanishes where
            # the injection is zero (natural gating); beta is added every step, i.e. tonic drive.
            # The beta half stays allocated either way so checkpoint shapes do not depend on mode.
            cur1 = cur1 * v1[:, 0]
            if self.film_mode == "gamma_beta":
                cur1 = cur1 + v1[:, 1]
        if gated:
            cur1 = gate * cur1
        mem1, syn1 = integrate(mem1, syn1, cur1, dec1,
                               self.alpha1_logit, syn_hidden)
        th1 = (self.thresh - self.v1 if self.bias_mode == "voltage"
               else self.thresh)
        spk1 = spike_fn(mem1 - th1, self.slope)
        if gated:
            spk1 = gate * spk1
        if self.hidden_reset == "zero":
            mem1 = mem1 * (1.0 - spk1)
        else:
            mem1 = mem1 - self.thresh * spk1

        # ---- sub-net layer 2: block diagonal ------------------------
        cur2 = torch.einsum("bgh,ghk->bgk", spk1, self.w2)
        if self.bias_mode == "current":
            cur2 = cur2 + self.b2
        if self.sub_ln in ("l2", "both"):
            cur2 = self.ln2(cur2)
        if self.sub_film in ("l2", "both"):
            v2 = self.film2(gait).view(-1, 2, G, Hg)
            cur2 = cur2 * v2[:, 0]
            if self.film_mode == "gamma_beta":
                cur2 = cur2 + v2[:, 1]
        if gated:
            cur2 = gate * cur2
        mem2, syn2 = integrate(mem2, syn2, cur2, dec2,
                               self.alpha2_logit, syn_hidden)
        th2 = (self.thresh - self.v2 if self.bias_mode == "voltage"
               else self.thresh)
        spk2 = spike_fn(mem2 - th2, self.slope)
        if gated:
            spk2 = gate * spk2
        if self.hidden_reset == "zero":
            mem2 = mem2 * (1.0 - spk2)
        else:
            mem2 = mem2 - self.thresh * spk2

        # ---- block-diagonal analog readout -------------------------
        curo = torch.einsum("bgh,ghk->bgk", spk2, self.w_read)
        if self.bias_mode == "current":
            curo = curo + self.b_read
        if gated:
            curo = gate * curo
        if self.readout_gait_bias:
            # AFTER the gate, deliberately: a DC offset, not an injection (gating it would make it
            # appear only on spike steps).
            curo = curo + self.b_read_gait(gait).view(-1, G, self.Ho)
        memo, syno = integrate(memo, syno, curo, deco,
                               self.alphao_logit, syn_readout)

        # `since` passes through untouched (kept only for the ONNX signature; see init_state).

        y_grp = torch.einsum("bgh,ghc->bgc", memo, self.w_out) + self.b_out
        y     = y_grp.flatten(1).index_select(1, self.out_perm)
        return (y, (mem_t, since, mem1, mem2, memo,
                    syn1, syn2, syno, mem_d),
                (spk_t,))

    def forward(self, x_seq, gait_seq, state=None, return_aux=False):
        """x_seq (L, B, n_neurons), gait_seq (L, B). Returns (y_seq, state), plus
        (spk_t_seq,) when return_aux=True.
        """
        B = x_seq.shape[1]
        if state is None:
            state = self.init_state(B, x_seq.device, x_seq.dtype)
        ys, at = [], []
        for t in range(x_seq.shape[0]):
            y, state, aux = self.step(x_seq[t], gait_seq[t], state)
            ys.append(y)
            if return_aux:
                at.append(aux[0])
        if return_aux:
            return torch.stack(ys), state, (torch.stack(at),)
        return torch.stack(ys), state

    @torch.no_grad()
    def timing_only(self, x_seq, gait_seq, mem_t=None):
        """(L, B, n_timing) timing spikes from CPG spikes via the eager `_timing`: exact,
        and cheap next to the sub-networks it skips.
        """
        if not self.uses_timing:
            # cpg / cpg_lif: the sub-networks are driven off the CPG's own channels (through the
            # drive LIF under cpg_lif), so return the CPG train unchanged. This keeps
            # timing_report and measure_rates working without a timing layer.
            return x_seq
        B = x_seq.shape[1]
        if mem_t is None:
            # n_timing, NOT G: this is the timing layer's own membrane.
            mem_t = torch.zeros(B, self.n_timing, device=x_seq.device,
                                dtype=x_seq.dtype)
        out = []
        for t in range(x_seq.shape[0]):
            spk, mem_t = self._timing(x_seq[t], gait_seq[t], mem_t)
            out.append(spk)
        return torch.stack(out)

    def param_breakdown(self):
        """Grouped parameter counts, for the startup print."""
        n = lambda *ps: int(sum(p.numel() for p in ps
                                 if p is not None))
        g = lambda name: getattr(self, name, None)
        # Under cpg / cpg_lif w1 is replaced by s_gate + u. s_gate is counted under "timing"
        # (it took over the per-gait routing job), keeping block totals comparable.
        return {
            "timing":  (n(self.w_in_gait.weight, self.beta_t_logit,
                          *(w.weight for w in (g("b_t"), g("v_t"))
                            if w is not None))
                        if self.uses_timing else n(self.s_gate.weight)),
            "sub_l1":  n(self.w1 if self.uses_timing else self.u,
                         self.beta1_logit, self.alpha1_logit,
                         g("b1"), g("v1")),
            "sub_l2":  n(self.w2, self.beta2_logit, self.alpha2_logit,
                         g("b2"), g("v2")),
            "readout": n(self.w_read, self.w_out, self.b_out,
                         self.betao_logit, self.alphao_logit, g("b_read"),
                         *( [self.b_read_gait.weight]
                            if self.readout_gait_bias else [] )),
            "sub_film": n(self.film1.weight, self.film2.weight),
        }



class SingleStepONNX(nn.Module):
    """Flat-signature wrapper so the exported graph is one timestep with
    explicit state in/out — the robot calls this once per CPG step."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, spikes, gait, mem1, mem2, memo):
        y, (m1, m2, mo), _ = self.model.step(
            spikes, gait, (mem1, mem2, memo))
        return y, m1, m2, mo


class SingleStepONNXTiming(nn.Module):
    """
    As SingleStepONNX but for TimingGroupedSNN's 9-tensor state.

    Written as an explicit signature rather than *state: torch.onnx.export
    traces varargs unreliably, and the input names have to line up
    positionally with `model.state_names_in` anyway.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, spikes, gait, mem_timing, since_upd, mem1, mem2, memo,
                syn1, syn2, syno, mem_drive):
        y, (mt, su, m1, m2, mo, s1, s2, so, md), _ = self.model.step(
            spikes, gait, (mem_timing, since_upd, mem1, mem2, memo,
                           syn1, syn2, syno, mem_drive))
        return y, mt, su, m1, m2, mo, s1, s2, so, md


def grad_blocks(model):
    """Group parameters into coarse blocks for per-block gradient reporting, by parameter
    NAME so it works for both architectures. Order matters: "w_in_gait" also startswith
    "w_in", so the timing block is tested first. Per block because a single |grad| hides
    which parts of the model get signal: a near-zero timing block means the timing layer
    is not learning, which a uniformly small gradient (harmless under Adam) would not.
    """
    rules = (
        # s_gate counts under "timing": under cpg / cpg_lif it took over the timing layer's
        # per-gait routing job, keeping block diagnostics comparable across modes.
        ("timing",    ("w_in_gait", "b_t", "beta_t_logit", "s_gate")),
        ("input",     ("w_in",)),                    # dense arch only
        ("sub_l1",    ("w1", "b1", "beta1_logit", "alpha1_logit", "u")),
        ("sub_l2",    ("w2", "b2", "beta2_logit", "alpha2_logit")),
        # "b_read" also prefix-matches "b_read_gait", which belongs here too.
        ("readout",   ("w_read", "b_read", "w_out", "b_out", "betao_logit",
                       "alphao_logit")),
        ("sub_film",  ("film1", "film2")),
        ("layernorm", ("ln1", "ln2")),
    )
    blocks = {}
    for pname, p in model.named_parameters():
        if not p.requires_grad:
            continue
        block = "other"
        for bname, prefixes in rules:
            if pname.startswith(prefixes):
                block = bname
                break
        blocks.setdefault(block, []).append(p)
    return blocks


def _tau_from_logit(logit):
    """Membrane/synapse time constant from a stored decay logit.
    Decays are stored as logit(beta) with beta = exp(-1/tau)."""
    d = torch.sigmoid(logit.detach().float()).clamp(1e-6, 1.0 - 1e-9)
    return (-1.0 / torch.log(d)).cpu().numpy()
