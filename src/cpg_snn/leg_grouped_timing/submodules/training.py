"""
Training: the stream sampler (truncated BPTT with carried state), spike objectives, gain
calibration and unit re-initialization, losses and the penalty schedule, metrics
logging, and the training loop.
"""

import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .common import out_path
from .model import _W1_INIT, _W_IN_INIT, grad_blocks


# ═══════════════════════════════════════════════════════════════════
# Stream sampler  (truncated BPTT with carried state)
# ═══════════════════════════════════════════════════════════════════

class StreamSampler:
    """B independent read heads walking the CPG spike train. Each holds its own position,
    active gait and countdown to the next gait switch. `next_chunk(L)` returns L
    consecutive steps per head; the caller carries (and detaches) the SNN state across
    calls, so training matches deployment: one continuous stream with the gait changing
    underneath. A head that runs off the end is rewound to a random start and flagged
    in `reset_mask` so the caller can zero its state.

    The whole dataset is tiny, so spikes, targets and valid live on the training device
    and batches are gathered on-device; the per-head gait timeline stays in numpy
    (branchy, and switches are rare).
    """

    def __init__(self, spikes, targets, valid, t_lo, t_hi, batch,
                 switch_min=600, switch_max=3000, rng=None, n_gaits=4,
                 device=None, phase=None, warm_steps=0):
        self.t_lo    = int(t_lo)
        self.t_hi    = int(t_hi)
        self.B       = int(batch)
        self.smin    = int(switch_min)
        self.smax    = int(switch_max)
        self.n_gaits = int(n_gaits)
        self.rng     = rng or np.random.default_rng(0)

        # One code path for CPU and GPU (device=None keeps everything on the CPU).
        self.device = torch.device(device) if device is not None \
            else torch.device("cpu")
        self.spikes  = torch.as_tensor(np.ascontiguousarray(spikes),
                                       dtype=torch.float32,
                                       device=self.device)
        self.targets = torch.as_tensor(np.ascontiguousarray(targets),
                                       dtype=torch.float32,
                                       device=self.device)
        self.valid   = torch.as_tensor(
            np.ascontiguousarray(valid).astype(np.float32),
            dtype=torch.float32, device=self.device)
        # NaNs zeroed here: `valid` is ~isnan(phase), so the mask already excludes those steps,
        # and NaNs in the loss would poison it via 0*NaN.
        ph = (np.zeros(len(spikes), dtype=np.float32) if phase is None
              else np.nan_to_num(np.asarray(phase, dtype=np.float32)))
        self.phase = torch.as_tensor(np.ascontiguousarray(ph),
                                     dtype=torch.float32, device=self.device)
        # Steps each head has advanced since its state was last zeroed (init_state() hands out zeros).
        self.warm_steps  = int(warm_steps)
        self.since_reset = np.zeros(self.B, dtype=np.int64)

        self.pos   = self.rng.integers(t_lo, t_hi, size=self.B)
        self.gait  = self.rng.integers(0, n_gaits, size=self.B)
        self.count = self.rng.integers(self.smin, self.smax, size=self.B)
        self._off  = {}                     # cached arange(L) per bptt

    def mark_all_reset(self):
        """Record that the network state of every head was just zeroed. Call whenever the
        caller re-initialiizes the model state outside the sampler (run_training does, at the
        start of every train and val pass): the warm-up mask is driven by `since_reset`,
        which the sampler otherwise only zeroes on its own rewinds. Without it the
        cold-start transient was scored every epoch, ~1/8 of an 8-chunk val, which is how
        val loss came out ~10x high under --synaptic.
        """
        self.since_reset[:] = 0

    def _offsets(self, L):
        off = self._off.get(L)
        if off is None:
            off = torch.arange(L, device=self.device, dtype=torch.long)
            self._off[L] = off
        return off

    def _rewind(self, b, L):
        hi = self.t_hi - L
        if hi <= self.t_lo:
            raise ValueError(
                f"stream range [{self.t_lo}, {self.t_hi}) is shorter than "
                f"bptt={L}; raise --tmax or lower --bptt/--val_frac.")
        self.pos[b]   = self.rng.integers(self.t_lo, hi)
        self.gait[b]  = self.rng.integers(0, self.n_gaits)
        self.count[b] = self.rng.integers(self.smin, self.smax)

    def next_chunk(self, L):
        B = self.B
        g   = np.zeros((L, B), dtype=np.int64)
        sw  = np.zeros((L, B), dtype=np.float32)   # 1 on the switch step
        reset_mask = np.zeros(B, dtype=np.float32)

        # ── per-head bookkeeping: rewind + gait timeline (CPU) ────
        for b in range(B):
            if self.pos[b] + L > self.t_hi:
                self._rewind(b, L)
                reset_mask[b] = 1.0
                self.since_reset[b] = 0

            gs = np.full(L, self.gait[b], dtype=np.int64)
            c  = self.count[b]
            while c < L:
                new_g = self.rng.integers(0, self.n_gaits)
                while new_g == gs[c] and self.n_gaits > 1:
                    new_g = self.rng.integers(0, self.n_gaits)
                gs[c:] = new_g
                sw[c, b] = 1.0
                c += self.rng.integers(self.smin, self.smax)
            self.gait[b]  = int(gs[-1])
            self.count[b] = int(c - L)
            g[:, b] = gs

        # Capture the starts AFTER any rewind, then advance.
        starts = self.pos.copy()
        self.pos = self.pos + L

        # ── on-device gather ──
        # idx[t, b] = absolute timestep head b refers to at position t; (1,B) + (L,1) broadcasts
        # to (L,B), and indexing a (T,4) tensor with it gives (L,B,4), the layout the loop wants.
        dev   = self.device
        pos_t = torch.as_tensor(starts, dtype=torch.long, device=dev)
        idx   = pos_t[None, :] + self._offsets(L)[:, None]      # (L, B)
        g_t   = torch.as_tensor(g, dtype=torch.long, device=dev)

        x = self.spikes[idx]                        # (L, B, n_cpg_neurons)
        y = self.targets[g_t, idx]                              # (L, B, J)
        m = self.valid[idx]                                     # (L, B)
        ph = self.phase[idx]                                    # (L, B)

        # ── post-reset warm-up mask ──
        # A head whose state was just zeroed cannot know where it is in the cycle, so its first
        # `warm_steps` outputs are not trained on (the forward pass still runs; only the loss is
        # masked). Heads reset two ways: the sampler's own rewinds, and mark_all_reset, which
        # covers EVERY head at the start of each train and val pass (without it the cold-start
        # transient was scored every epoch).
        warm = np.ones((L, B), dtype=np.float32)
        if self.warm_steps > 0:
            for b in range(B):
                k = max(0, min(L, self.warm_steps - int(self.since_reset[b])))
                if k > 0:
                    warm[:k, b] = 0.0
        self.since_reset += L

        return (x, g_t, y, m,
                torch.as_tensor(sw, dtype=torch.float32, device=dev),
                torch.as_tensor(reset_mask, dtype=torch.float32, device=dev),
                ph,
                torch.as_tensor(warm, dtype=torch.float32, device=dev))

# Calibration target bands, as FRACTIONS OF THE PERIOD (timesteps per cycle), not of the
# spike-count ceiling (n_cpg * cpg_rate): what matters is how much of the cycle, in time,
# carries input, and --cpg_refrac_main lengthens the period without changing the ceiling.
# The saturation cap stays on the ceiling (a hard count limit); see the main() calibration
# block. At the default refrac_main=1 (ISI 2) period == 2 * ceiling, giving 15-20 spk/cyc
# (plain) and 20-40 (min_count) at n_cpg=6.
_BAND_FRAC_PLAIN     = (0.125, 1.0 / 6.0)       # 15-20 spk/cyc at n_cpg=6, ISI=2
_BAND_FRAC_MIN_COUNT = (1.0 / 6.0, 1.0 / 3.0)   # 20-40 spk/cyc at n_cpg=6, ISI=2

# Highest fraction of the phase-blind ceiling (n_cpg * cpg_rate, where a unit's train is
# the OR of the CPG's and carries no phase) that calibration may target.
_SAT_FRAC = 0.6


# ═══════════════════════════════════════════════════════════════════
# Keeping the small spiking layers alive
# ═══════════════════════════════════════════════════════════════════
# The timing layer has no LayerNorm, so nothing holds its input current where a threshold-1.0
# LIF fires. Three mechanisms: calibrate_gains (once, before training: never start dead or
# loud), SpikeObjective (every step: shapes the spike train) and reinit_timing_units (on
# detection: rescues what failed). Both extremes are fatal: a dead timing unit gives the w1
# slots it feeds EXACTLY zero gradient, and a saturated one fires on every CPG spike (its
# train is the CPG's OR, no phase) while sitting where the surrogate gradient is ~1e-6, which
# Adam's scale-invariance means a larger --spike_stats_lambda cannot fix.


@torch.no_grad()
def measure_rates(model, spikes, n_gaits, device,
                  t0=0, n_steps=None, period=254.0):
    """Spikes per cycle for every (gait, timing unit), replaying the CPG at batch 1.
    Returns (n_gaits, n_timing); forward passes only.
    """
    if n_steps is None:
        n_steps = int(min(20 * period, len(spikes) - t0))
    n_steps = int(min(n_steps, len(spikes) - t0))
    x = torch.as_tensor(spikes[t0:t0 + n_steps], dtype=torch.float32,
                        device=device).unsqueeze(1)
    n_cycles = max(n_steps / float(period), 1e-9)

    out = []
    for g in range(n_gaits):
        gg = torch.full((n_steps, 1), g, dtype=torch.long, device=device)
        spk = model.timing_only(x, gg)
        out.append((spk[:, 0].sum(0) / n_cycles).cpu().numpy())
    return np.stack(out)


@torch.no_grad()
def calibrate_gains(model, spikes, n_gaits, device, period,
                    lo=1.0, hi=5.0, iters=18, g_lo=1e-3, g_hi=1e3,
                    per_gait=True, cpg_rate_hint=None, verbose=True):
    """Scale each timing unit's CPG->timing weight column so its firing rate starts
    inside [lo, hi] spikes per cycle, by bisection on measured rates (no user scale
    factor). Writes into `w_in_gait`.

    Rate is monotone non-decreasing in a unit's own gain (a step function: it jumps
    by whole spikes), hence a band rather than an exact target. Units are independent
    (unit l's current depends only on column l of each gait's matrix), so all bisect
    in parallel.

    Weights, not bias: a bias is tonic drive that also exists during CPG silence, so
    raising it makes a unit alive but unrelated to the rhythm; scaling weights
    amplifies the phase preference it already has. Gain cannot fix a wrong sign (a
    net-negative unit rides to g_hi and gives up; the positive-mean init prevents
    that); such units are reported.

    per_gait=True (default) bisects a separate gain per (gait, unit). per_gait=False
    is the old one-gain-per-unit behavior, bisected against the MINIMUM rate across
    gaits, which is a saturation generator: nothing bounded the maximum, so one
    weakly driven gait dragged the others up (max rates of 5-48 spk/cyc against a
    [1, 5] band over 24 draws). Training cannot undo that: a unit at the ceiling sits
    where the surrogate gradient is ~1e-6, and Adam's scale-invariance means raising
    --spike_stats_lambda 10,000x moved the gain 1.5%. Rows of w_in_gait past n_gaits
    are left at gain 1.

    `cpg_rate_hint` (the CPG's spikes per cycle, from cpg_spike_stats) only feeds the
    saturation warning: a unit can reach at most n_cpg * cpg_rate, one spike per CPG
    spike.
    """
    if not hasattr(model, "w_in_gait"):
        return {}

    # G here is the timing-unit count (model.n_timing, not model.G); w_in_gait is indexed
    # by timing unit.
    MG, n_cpg, G = model.max_gaits, model.n_neurons, model.n_timing
    # (max_gaits, n_cpg, n_timing) view; column l is timing unit l's inputs.
    W = model.w_in_gait.weight.data.view(MG, n_cpg, G)
    base = W.clone()

    # Bisection brackets: one per (gait, unit) in per-gait mode, one per unit otherwise;
    # everything below is written against `shape` so both modes share one loop.
    shape = (n_gaits, G) if per_gait else (G,)
    lo_g = torch.full(shape, g_lo, device=device)
    hi_g = torch.full(shape, g_hi, device=device)

    def rates_at(gain):
        """Write `gain` into w_in_gait and measure; returns a tensor shaped like `gain`."""
        if per_gait:
            # (n_gaits, G) -> (max_gaits, 1, G); rows past n_gaits are never indexed, so stay at 1.
            full = torch.ones(MG, G, device=device, dtype=base.dtype)
            full[:n_gaits] = gain
            W.copy_(base * full.unsqueeze(1))
        else:
            # gain (G,) broadcasts over (max_gaits, n_cpg, G): each unit's whole column, every gait.
            W.copy_(base * gain.view(1, 1, G))
        r = torch.as_tensor(
            measure_rates(model, spikes, n_gaits, device, period=period),
            device=device)                                  # (n_gaits, G)
        return r if per_gait else r.min(dim=0).values

    for _ in range(iters):
        mid = torch.sqrt(lo_g * hi_g)                   # geometric midpoint
        r   = rates_at(mid)
        too_quiet = r < lo
        too_loud  = r > hi
        lo_g = torch.where(too_quiet, mid, lo_g)
        hi_g = torch.where(too_loud,  mid, hi_g)
        if not (too_quiet | too_loud).any():
            break

    # sqrt(lo*hi) is safe to return: once a unit is in band neither bound updates, so its
    # midpoint is frozen at the value measured in band.
    final_gain  = torch.sqrt(lo_g * hi_g)
    final_rates = rates_at(final_gain)
    ok = ((final_rates >= lo * 0.5) & (final_rates <= hi * 2.0))

    # rate_by_gait is always the full (n_gaits, G) measurement, so both modes stay
    # comparable in the config dump.
    rate_by_gait = final_rates if per_gait else torch.as_tensor(
        measure_rates(model, spikes, n_gaits, device, period=period),
        device=device)

    tolist = lambda t: t.cpu().tolist()
    report = {"timing": {
        "per_gait":     bool(per_gait),
        "band":         [float(lo), float(hi)],
        "gain":         tolist(final_gain),      # (n_gaits, G) or (G,) by mode
        "rate_by_gait": tolist(rate_by_gait),    # always (n_gaits, G)
        # Original names and meanings kept so older configs still read; max_rate is the one to watch.
        "min_rate":     tolist(rate_by_gait.min(dim=0).values),
        "max_rate":     tolist(rate_by_gait.max(dim=0).values),
        "in_band":      tolist(ok),
    }}

    if verbose:
        fmt  = lambda v: " ".join(f"{x:6.2f}" for x in v)
        tim  = report["timing"]
        mode = "per (gait, unit)" if per_gait else "per unit, min over gaits"
        print(f"      calibrated timing [{mode}], band [{lo}, {hi}] spk/cyc")
        if per_gait:
            for g in range(n_gaits):
                print(f"        gait {g}: gain [{fmt(tim['gain'][g])}]")
                print(f"                 rate [{fmt(tim['rate_by_gait'][g])}]")
        else:
            print(f"        gain     [{fmt(tim['gain'])}]")
        print(f"        min spk/cyc across gaits [{fmt(tim['min_rate'])}]")
        print(f"        MAX spk/cyc across gaits [{fmt(tim['max_rate'])}]")

        # A unit at the ceiling fires once per CPG spike (the CPG's OR, no phase). The threshold is
        # _SAT_FRAC, not looser: 0.9 would have missed a (gait, unit) pair at 86% of the ceiling.
        ceiling = float(n_cpg) * float(cpg_rate_hint) if cpg_rate_hint else None
        if ceiling:
            sat = (rate_by_gait >= _SAT_FRAC * ceiling).nonzero().cpu().tolist()
            if sat:
                print(f"      WARNING: (gait, unit) {sat} are at >={_SAT_FRAC:g} x the "
                f"{ceiling:.0f} spk/cyc ceiling, so they carry little or no phase, and "
                f"training will not fix it (surrogate gradient ~1e-6).")

        bad = ok.logical_not().nonzero().cpu().tolist()
        if bad:
            label = "(gait, unit)" if per_gait else "unit"
            print(f"      WARNING: timing {label} {bad} could not be brought into "
            f"[{lo}, {hi}] spk/cyc by scaling alone (likely net-negative input "
            f"current); if they stay dead, check the _W_IN_INIT positive mean.")
    return report


# ── Spike-statistics objectives (strategy pattern) ─────────
# What the spike train SHOULD look like is open, so the objective is swappable: add a
# subclass with a `name` and it becomes --spike_objective <name>. All share one signature,
# so run_training never branches:
#     penalty(spk, gait, phase, mask) -> scalar
#     spk (L, B, U) surrogate spikes from step()'s aux;  gait (L, B) int64;
#     phase (L, B) in [0,1), NaNs zeroed;  mask (L, B) 1 where the sample counts
#     (valid phase AND past the post-reset warm-up).
# Statistics are grouped per (gait, unit), never averaged over the batch (that would hide a
# unit misbehaving for one gait); gaits absent from a chunk are masked out, not counted silent.


class SpikeObjective:
    """Base class.  Subclasses implement `penalty`; `lam == 0` disables."""

    name = "base"

    def __init__(self, lam=0.0, period=254.0, n_gaits=4,
                 target_rate=None, target_R=None,
                 min_count_floor=0.0, floor_weight=1.0):
        self.lam        = float(lam)
        self.period     = float(period)
        self.n_gaits    = int(n_gaits)
        self.target_rate = target_rate     # CPG spikes per cycle
        self.target_R    = target_R        # CPG circular concentration
        self.min_count_floor = float(min_count_floor)   # spikes/cycle
        self.floor_weight    = float(floor_weight)

    def _grouped(self, spk, gait, mask):
        """(rate, present, cnt) per (gait, unit).  rate is per-timestep."""
        oh   = torch.nn.functional.one_hot(gait, self.n_gaits).to(spk.dtype)
        oh   = oh * mask.unsqueeze(-1)                        # (L,B,G)
        cnt  = torch.einsum("lbg,lbu->gu", oh, spk)
        den  = oh.sum(dim=(0, 1))
        rate = cnt / den.clamp(min=1.0).unsqueeze(1)
        present = (den > 0).to(spk.dtype).unsqueeze(1)
        return rate, present, cnt, oh

    @property
    def enabled(self):
        return self.lam > 0.0

    def penalty(self, spk, gait, phase, mask):
        raise NotImplementedError

    def describe(self):
        return f"{self.name} (lam={self.lam:g})"


class NoSpikeObjective(SpikeObjective):
    """No constraint on the timing layer's spike statistics."""

    name = "none"

    @property
    def enabled(self):
        return False

    def penalty(self, spk, gait, phase, mask):
        return spk.sum() * 0.0

    def describe(self):
        return "none (timing spike statistics unconstrained)"


class CPGMatchSpikeObjective(SpikeObjective):
    """Fire LIKE THE CPG: match its spikes per cycle and burst tightness. Two terms,
    because rate alone cannot tell one tight burst from a lone spike every 35 steps:
      rate  ((rate - target)/target)^2, two-sided, relative so lam is scale-free.
      conc  relu(target_R - R)^2, one-sided; R = |sum_t spk_t exp(i 2pi phase_t)| /
            sum_t spk_t is the circular concentration of the unit's spike phases (near
            1 = one burst per cycle). `phase` is constant and `spk` carries the
            surrogate gradient, so it is differentiable as written.
    Limitation: R uses the fundamental only, so two bursts at opposite phases cancel to
    R ~ 0 (wrong for a genuinely two-burst gait). NOT compatible with gated
    sub-networks, which need spikes wherever the output changes. Kept for comparison.
    """

    name = "cpg_match"

    def penalty(self, spk, gait, phase, mask):
        rate, present, cnt, oh = self._grouped(spk, gait, mask)
        tgt = self.target_rate / self.period

        rate_err = ((rate - tgt) / tgt) ** 2 * present

        ang = 2.0 * math.pi * phase
        C = torch.einsum("lbg,lbu,lb->gu", oh, spk, torch.cos(ang))
        S = torch.einsum("lbg,lbu,lb->gu", oh, spk, torch.sin(ang))
        R = torch.sqrt(C ** 2 + S ** 2 + 1e-8) / cnt.clamp(min=1e-2)
        # Only shape units that are already firing (R is meaningless for a silent unit; the rate
        # term pushes it up). Detached hard mask.
        alive = (cnt > 1.0).to(spk.dtype).detach()
        conc_err = torch.relu(self.target_R - R) ** 2 * present * alive

        denom = present.sum().clamp(min=1.0) * spk.shape[2]
        return self.lam * (rate_err.sum() + conc_err.sum()) / denom

    def describe(self):
        return (f"cpg_match (lam={self.lam:g}, target "
                f"{self.target_rate:.2f} spk/cyc, R={self.target_R:.3f})")


class MinCountSpikeObjective(SpikeObjective):
    """Spend as few spikes as possible: a flat cost per spike beyond a free allowance,
    with the TASK loss deciding where the rest are worth spending. Needs the output to
    depend on the spike train alone (natural or explicit gating; see TimingGroupedSNN).
    Then the cheapest solution places spikes densely where the target changes fast and
    sparsely where it creeps: an adaptive sampling clock derived from the data.

    Linear in the rate, not squared: a flat marginal cost per spike is the L1-sparsity
    form, while a squared cost gives up as the rate falls.

    FLOOR (`min_count_floor`, spikes per CYCLE), a two-sided hinge:
        penalty = lam * (relu(duty - floor/period)
                         + floor_weight * relu(floor/period - duty))
    Above the floor it is L1 (marginal cost lam per spike); below it the gradient
    reverses, so descent raises the rate. It has to be a hinge: lam*(duty - floor/period)
    has the same gradient as lam*duty (the derivative is constant), so only a kink can
    change the gradient's sign.

    Why a floor: without one the penalty keeps pushing to zero, and a silent unit is
    UNRECOVERABLE through the task loss (it passes exactly zero gradient to the w1 slots
    it feeds); one run pruned several sub-networks to an excellent 2-5 spikes per cycle
    and others to 0. The floor term still reaches a silent unit because it acts on the
    timing membrane through spike_fn's small non-zero surrogate (~0.03 at
    timing_slope=5). Cost: the floor is an attractor (a unit the task would run at 1
    spk/cyc is pushed up to it), so set it to the fewest spikes a sub-network can use.
    floor_weight=0 gives the one-sided version; min_count_floor=0 the original L1.

    It does NOT fix saturation (surrogate gradient ~1e-6, and Adam's scale-invariance
    means a larger lam cannot compensate); reinit_timing_units handles that. See also
    --lambda_hold / --lambda_warmup and --reinit_dead_after.
    """

    name = "min_count"

    def penalty(self, spk, gait, phase, mask):
        rate, present, cnt, oh = self._grouped(spk, gait, mask)
        # rate is per-TIMESTEP duty; the floor is per CYCLE, so convert.
        floor_duty = self.min_count_floor / self.period
        over  = torch.relu(rate - floor_duty)
        under = torch.relu(floor_duty - rate)
        term  = over + self.floor_weight * under
        denom = present.sum().clamp(min=1.0) * spk.shape[2]
        return self.lam * (term * present).sum() / denom

    def describe(self):
        return (f"min_count (lam={self.lam:g}, linear/L1 in duty cycle above "
                f"a floor of {self.min_count_floor:g} spk/cyc, "
                f"floor_weight={self.floor_weight:g})")


SPIKE_OBJECTIVES = {cls.name: cls for cls in (
    NoSpikeObjective, CPGMatchSpikeObjective, MinCountSpikeObjective)}


def make_spike_objective(name, **ctx):
    if name not in SPIKE_OBJECTIVES:
        raise ValueError(f"Unknown --spike_objective {name!r}; "
                         f"available: {sorted(SPIKE_OBJECTIVES)}")
    return SPIKE_OBJECTIVES[name](**ctx)


@torch.no_grad()
def reinit_timing_units(model, dead_pairs, sat_pairs, n_gaits,
                        generator=None, sat_scale=0.5, verbose=True):
    """Rescue failed timing units, PER (gait, unit). `dead_pairs` / `sat_pairs` are lists
    of (gait, unit) tuples. Per-gait because w_in_gait already is: gait g's column for
    unit u is an independent slice, so fixing it cannot disturb any other gait, unit or
    anything downstream.

    DEAD (0 spk/cyc): re-roll that (gait, unit) column with the positive mean restored
        and zero its per-gait bias (a silent column carries no direction worth keeping).
    SATURATED: SCALE the column down by `sat_scale` instead. This is a magnitude
        failure, not a direction one: the phase preference is still in the column, just
        swamped, and rate is monotone in gain, so halving strictly reduces it (a badly
        saturated unit may take several triggers).
    Neither can be left to training (zero gradient when silent, ~1e-6 when saturated).

    `w1` is re-rolled only for units dead in EVERY gait: d(cur1)/d(w1) is proportional
    to the timing spike, so those slots stayed at random init while the sub-network
    trained around them, whereas a unit dead in only some gaits trained its slots on the
    others. w1 is indexed by sub-network, so the slots re-rolled are the (g, k) with
    timing_map[g][k] == u; that sub-network's other K-1 slots are left alone.
    w2/w_read/w_out are never touched (they trained and may hold useful output range).
    """
    if not (dead_pairs or sat_pairs) or not hasattr(model, "w_in_gait"):
        return
    dev = model.w_in_gait.weight.device
    MG, nn_, T = model.max_gaits, model.n_neurons, model.n_timing
    W = model.w_in_gait.weight.view(MG, nn_, T)

    for g, u in dead_pairs:
        W[g, :, u] = (torch.randn(nn_, generator=generator).to(dev)
                      * _W_IN_INIT + _W_IN_INIT)
        # Per-gait excitability back to neutral, in whichever form is in use ("voltage" mode has
        # v_t and no b_t; "none" has neither).
        for name in ("b_t", "v_t"):
            emb = getattr(model, name, None)
            if emb is not None:
                emb.weight[g, u] = 0.0

    for g, u in sat_pairs:
        W[g, :, u] *= sat_scale

    # w1 only for units dead in every gait -- see the docstring.
    dead_by_unit = {}
    for g, u in dead_pairs:
        dead_by_unit.setdefault(u, set()).add(g)
    everywhere = sorted(u for u, gs in dead_by_unit.items()
                        if len(gs) >= n_gaits)
    slots = {}
    if everywhere:
        K = model.K
        scale = _W1_INIT / math.sqrt(K)      # same as the constructor's init
        for u in everywhere:
            hit = [(g, k) for g, m in enumerate(model.timing_map_list)
                   for k, t in enumerate(m) if t == u]
            for g, k in hit:
                model.w1[g, k] = (torch.randn(model.Hg, generator=generator)
                                  .to(dev) * scale)
            slots[u] = hit

    if verbose:
        if dead_pairs:
            print(f"      [reinit] DEAD (gait, unit) {sorted(dead_pairs)}: "
                  f"re-rolled that gait's w_in_gait column and zeroed its "
                  f"per-gait bias")
        if everywhere:
            print(f"      [reinit] unit(s) {everywhere} were dead for ALL "
                  f"{n_gaits} gaits, so sub-net w1 slot(s) {slots} were "
                  f"re-rolled too (they had received zero gradient throughout)")
        if sat_pairs:
            print(f"      [reinit] SATURATED (gait, unit) "
                  f"{sorted(sat_pairs)}: scaled that gait's w_in_gait column "
                  f"by {sat_scale:g} (magnitude failure, so the column's "
                  f"direction is kept)")


# ═══════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════

def make_gait_weights(tables_orig, names, device):
    """Upweight gaits whose original table was coarsest (bk: 22 rows). `names` is required
    (not defaulted) so a silent fallback cannot print the wrong labels next to the right
    numbers.
    """
    R = max(t.shape[0] for t in tables_orig)
    w = torch.tensor([R / t.shape[0] for t in tables_orig],
                     dtype=torch.float32, device=device)
    print("      gait loss weights: " +
          "  ".join(f"{names[i]}={w[i].item():.2f}" for i in range(len(w))))
    return w


def masked_loss(pred, targ, mask, gait, gait_w=None):
    """MSE over (L,B,8), masked by (L,B), optionally weighted per gait."""
    err = ((pred - targ) ** 2).mean(dim=2)          # (L,B)
    if gait_w is not None:
        err = err * gait_w[gait]
    denom = mask.sum().clamp(min=1.0)
    return (err * mask).sum() / denom


def detach_state(state):
    return tuple(s.detach() for s in state)


def apply_reset(state, reset_mask):
    """Zero the state of any stream that was rewound. `reset_mask` is (B,); it is reshaped
    to match each state tensor's own rank, not a hardcoded one: a wrong rank broadcasts
    instead of failing (a (B,1,1) mask on (B,H) state silently gives (B,B,H)), and
    TimingGroupedSNN's state mixes ranks (mem_timing is (B, n_timing); mem1/mem2/memo are
    (B, G, Hg)).
    """
    if reset_mask.sum() == 0:
        return state
    keep = 1.0 - reset_mask
    return tuple(s * keep.view(-1, *([1] * (s.dim() - 1))) for s in state)


class MetricsWriter:
    """Append one row per epoch to out_dir/metrics.csv, flushed every epoch (Ctrl+C is a
    normal way to finish a run, and the config's `history` only lands on a clean exit).
    Columns are fixed from the first row; a key appearing later is dropped rather than
    shifting the table.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.cols = None
        self.fh = None

    def write(self, row):
        if self.fh is None:
            self.cols = list(row.keys())
            self.fh = open(self.path, "w")
            self.fh.write(",".join(self.cols) + "\n")
        vals = []
        for c in self.cols:
            v = row.get(c, "")
            vals.append("" if v is None else
                        (f"{v:.6g}" if isinstance(v, float) else str(v)))
        self.fh.write(",".join(vals) + "\n")
        self.fh.flush()

    def close(self):
        if self.fh is not None:
            self.fh.close()
            self.fh = None


def lambda_schedule(epoch, hold, warmup):
    """Sparsity-penalty multiplier for a 1-based epoch: 0.0 through `hold`, then
    (epoch - hold) / warmup, then 1.0. hold=0 is the old ramp-from-epoch-1; warmup=0 is a
    step at hold+1.
    """
    if epoch <= hold:
        return 0.0
    if warmup <= 0:
        return 1.0
    return min(1.0, (epoch - hold) / float(warmup))


def lambda_schedule_text(args):
    """One-line human description of the schedule, for startup prints."""
    h, w = int(args.lambda_hold), int(args.lambda_warmup)
    parts = []
    if h > 0:
        parts.append(f"0 for epochs 1-{h}")
    if w > 0:
        parts.append(f"linear ramp over epochs {h + 1}-{h + w}")
    parts.append(f"full strength from epoch {h + w + 1}")
    return ", ".join(parts)


def run_training(model, tr_sampler, va_sampler, opt, sched, device, args,
                 gait_w, out_dir, timing_diag=None, n_gaits=4, period=254.0,
                 spike_obj=None, sat_rate=None, tgt_range=None,
                 gait_names=None):
    """`timing_diag`: optional zero-arg callable returning (lines, stats), called every
        args.timing_log_every epochs for timing_grouped (None for dense); its last return
        value is handed back so the config can record it.
    `n_gaits`, `period`: reporting and diagnostics.
    `spike_obj`: a SpikeObjective whose `penalty` is added to the task loss
        (NoSpikeObjective disables it).
    `sat_rate`: spk/cyc at or above which a timing unit counts as saturated, for the
        reinit trigger; the same value passed to timing_report, so the warning and the
        action cannot disagree. None disables saturation rescue.
    """
    best = float("inf")
    best_path = out_path(out_dir, "best_model.pt")
    hist = {"train": [], "val": [], "val_sw": [], "gnorm": [], "sec": [],
            "floor": [], "upd": []}
    last_timing_stats = []

    # Per-block gradient norms and the per-epoch update norm. |upd| measures movement directly:
    # under Adam the update is scale-invariant in the gradient, so a small |grad| says nothing
    # about whether the model is moving. Expected ~lr*sqrt(N) per gradient step.
    blocks = grad_blocks(model)
    for b in blocks:
        hist[f"g_{b}"] = []
        hist[f"u_{b}"] = []
    n_par = sum(p.numel() for p in model.parameters())
    exp_upd = args.lr * math.sqrt(n_par) * math.sqrt(args.chunks_per_epoch)
    metrics = MetricsWriter(out_path(out_dir, "metrics.csv"))
    print(f"\n  Per-epoch metrics -> {out_dir / 'metrics.csv'}")
    print(f"  Gradient blocks: {', '.join(sorted(blocks))}")
    print(f"  |upd| = per-epoch ||delta theta||. Order-of-magnitude "
          f"expectation at lr={args.lr:g} is ~{exp_upd:.2f}")
    print(f"  (Adam's step is scale-invariant in the gradient, so |upd| -- not "
          f"|grad| -- is what says whether the model is moving.)")
    print(f"  u/g per block is the diagnostic: Adam normalizes PER PARAMETER, so")
    print(f"  a block moves at ~lr per weight no matter how small its gradient.")
    print(f"  u/g >> 1 means a block is moving far more than its share of the")
    print(f"  signal warrants -- i.e. random-walking on noise, which perturbs")
    print(f"  the representation the downstream layers are trying to use.")

    # Spike statistics are only meaningful for the arch with a timing layer.
    use_stats = (spike_obj is not None and spike_obj.enabled
                 and hasattr(model, "n_timing"))
    if use_stats:
        print(f"\n  Spike objective: {spike_obj.describe()}")
        if args.lambda_hold > 0 or args.lambda_warmup > 0:
            print(f"  lambda: {lambda_schedule_text(args)}, so the network "
                  f"learns to USE spikes before it is charged for them.")
        print(f"  Judge it on free-run RMSE, not on the spike count.")
    # Announced separately: under cpg / cpg_lif the spike objective is forced off, yet the
    # same schedule applies to the drive penalty.
    if args.drive_lambda > 0.0 and (args.lambda_hold > 0
                                    or args.lambda_warmup > 0):
        print(f"\n  Drive penalty ({args.drive_penalty}, lambda "
              f"{args.drive_lambda:g}): {lambda_schedule_text(args)}, so the "
              f"task can show which channels matter before the penalty "
              f"starts pruning.")

    # Consecutive timing reports each (gait, unit) pair has been dead or saturated, for
    # reinit_timing_units. Keyed per pair (the rescue is per-gait); advanced only on diagnostic epochs.
    dead_streak, sat_streak = {}, {}

    print(f"\n  {'Epoch':>6}  {'Train':>10}  {'Val':>10}  "
          f"{'Val(post-sw)':>13}  {'LR':>9}  {'|grad|':>8}  {'|upd|':>8}"
          f"  {'sec':>6}")
    print("  " + "-" * 88)
    print(f"  (Ctrl+C stops training and exports.  |grad| is the PRE-clip "
          f"norm, clip={args.clip:g};")
    print(f"   routinely much larger than clip means most updates are being "
          f"truncated and the LR is too hot.)")

    # epoch0.pt: the model before the first update (after gain calibration), the only exact
    # record of the initial weights and taus.
    if args.save_model_every > 0:
        torch.save(model.state_dict(), out_path(out_dir, "epoch0.pt"))
        print("  [saved] epoch0.pt (initial weights, before any update)")

    try:
        for epoch in range(1, args.epochs + 1):
            # Times train + validate, the cost that recurs every epoch; read before the timing report so
            # a diagnostic epoch does not spike this column.
            t_epoch = time.perf_counter()
            # Per-block snapshot of the params, so the update norm can be attributed per block.
            theta0 = {b: [p.detach().clone() for p in plist]
                      for b, plist in blocks.items()}
            bacc = {b: torch.zeros((), device=device) for b in blocks}

            # ---- train -------------------------------------------------
            model.train()
            state = model.init_state(args.batch, device)
            tr_sampler.mark_all_reset()     # so the warm-up mask covers it
            # ONE schedule for every sparsity term (see lambda_schedule): both the spike objective and
            # the drive penalty can lock in a structure before the task has said what it needs.
            lam_scale = lambda_schedule(epoch, args.lambda_hold,
                                        args.lambda_warmup)
            tot, gtot, nb, ftot = 0.0, 0.0, 0, 0.0
            for _ in range(args.chunks_per_epoch):
                (x, g, y, m, sw, rst,
                 ph, warm) = tr_sampler.next_chunk(args.bptt)
                x, g, y, m = (x.to(device), g.to(device),
                              y.to(device), m.to(device))
                ph, warm = ph.to(device), warm.to(device)
                state = apply_reset(detach_state(state), rst.to(device))

                # `warm` zeroes steps whose head was reset less than one cycle ago: the forward pass still
                # runs (that is how state builds), but a zero-state network cannot know where it is in the
                # cycle, so those outputs are not trained against.
                m_eff = m * warm

                if use_stats:
                    pred, state, aux = model(x, g, state, return_aux=True)
                else:
                    pred, state = model(x, g, state)
                    aux = None
                loss = masked_loss(pred, y, m_eff, g, gait_w)

                if aux is not None:
                    # aux = (timing spikes,), each (L, B, U). Same warm mask as the task loss: a just-reset unit
                    # legitimately fires less.
                    pen = lam_scale * sum(
                        spike_obj.penalty(a, g, ph, m_eff) for a in aux)
                    loss = loss + pen
                    ftot += float(pen.detach())

                # Drive sparsity penalty (cpg / cpg_lif; participation ratio, or L1 with --drive_penalty l1) on
                # the shared hold/ramp schedule. Separate from the spike objective: it penaliizes the drive's
                # GAINS (under cpg the spikes are the CPG's own, an input; under cpg_lif the spike count is a
                # staircase in the gain). Folded into the same `floor` metric, so one column tracks whichever
                # pressure is active.
                if (args.drive_lambda > 0.0
                        and hasattr(model, "drive_penalty")):
                    dpen = (lam_scale * args.drive_lambda
                            * model.drive_penalty(args.drive_penalty))
                    loss = loss + dpen
                    ftot += float(dpen.detach())

                opt.zero_grad()
                loss.backward()
                # Returns the total norm BEFORE clipping, the only way to see whether clip=1.0 is quietly
                # truncating nearly every update (masking a too-hot LR). Taken before clip_grad_norm_, which
                # rescales in place; accumulated on device and synced once per epoch.
                with torch.no_grad():
                    for b, plist in blocks.items():
                        gs = [p.grad.detach() for p in plist
                              if p.grad is not None]
                        if gs:
                            bacc[b] += torch.linalg.vector_norm(
                                torch.stack([torch.linalg.vector_norm(g)
                                             for g in gs]))
                gnorm = nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                opt.step()
                # Keep voltage-mode thresholds strictly positive (see clamp_bias_voltage). getattr because
                # dense has no such parameters; `model` is the raw module (compile swaps `step`, not it).
                clamp = getattr(model, "clamp_bias_voltage", None)
                if clamp is not None:
                    clamp()
                # Stepped per GRADIENT STEP so the LR schedule does not depend on chunks_per_epoch
                # (T_max = epochs * chunks_per_epoch).
                sched.step()
                tot += loss.item(); gtot += float(gnorm); nb += 1
            tr_loss  = tot / max(nb, 1)
            tr_gnorm = gtot / max(nb, 1)
            tr_floor = ftot / max(nb, 1)

            # ---- validate ----------------------------------------------
            model.eval()
            vstate = model.init_state(args.batch, device)
            va_sampler.mark_all_reset()     # so the warm-up mask covers it
            # Accumulated as (masked error sum, valid-step count), not an average of chunk means: after the
            # per-epoch reset the first chunk is almost entirely masked, so an equal-weight mean would
            # count its few valid steps as a full chunk (and a fully masked chunk would drag val down).
            # masked_loss * mask.sum() recovers the exact masked sum.
            vtot, vsw_tot, vn, vsw_n = 0.0, 0.0, 0.0, 0.0
            # Per (gait, joint) squared error and step count for an RMSE in degrees, taken from this
            # pass's predictions (no extra forward pass).
            rm_sse = rm_cnt = None
            with torch.no_grad():
                for _ in range(args.val_chunks):
                    (x, g, y, m, sw, rst,
                     ph, warm) = va_sampler.next_chunk(args.bptt)
                    x, g, y, m = (x.to(device), g.to(device),
                                  y.to(device), m.to(device))
                    sw, warm = sw.to(device), warm.to(device)
                    vstate = apply_reset(vstate, rst.to(device))
                    pred, vstate = model(x, g, vstate)
                    m = m * warm            # same warm-up exclusion as train

                    nv = float(m.sum())
                    if nv > 0:
                        vtot += masked_loss(pred, y, m, g).item() * nv
                        vn += nv
                        if rm_sse is None:
                            rm_sse = torch.zeros(n_gaits, pred.shape[2],
                                                 device=pred.device)
                            rm_cnt = torch.zeros(n_gaits, device=pred.device)
                        e2 = ((pred - y) ** 2) * m.unsqueeze(-1)   # (L,B,J)
                        gi_ = g.reshape(-1).long()
                        rm_sse.index_add_(0, gi_, e2.reshape(-1, e2.shape[2]))
                        rm_cnt.index_add_(0, gi_, m.reshape(-1))

                    # "post-switch" window: args.settle steps after each switch
                    post = torch.zeros_like(sw)
                    idx = sw.nonzero(as_tuple=False)
                    for t_i, b_i in idx:
                        post[t_i:min(t_i + args.settle, sw.shape[0]), b_i] = 1.0
                    mp = m * post
                    npv = float(mp.sum())
                    if npv > 0:
                        vsw_tot += masked_loss(pred, y, mp, g).item() * npv
                        vsw_n   += npv

            if vn <= 0:
                raise RuntimeError(
                    f"validation scored ZERO steps: the post-reset warm-up "
                    f"({args.val_chunks} chunks x {args.bptt} steps) masked "
                    f"everything. Raise --val_chunks.")
            va_loss = vtot / vn
            vsw     = vsw_tot / vsw_n if vsw_n > 0 else float("nan")

            # Validation RMSE in degrees per gait (mean over joints of each joint's RMSE); targets use one
            # global (lo, hi), so the conversion is (hi-lo)/2. Measured on the validation STREAM, so gait
            # switches make it run above a steady single-gait RMSE.
            rmse_g = np.full(n_gaits, np.nan)
            if rm_sse is not None and tgt_range is not None:
                scale = (float(tgt_range[1]) - float(tgt_range[0])) / 2.0
                cnt = rm_cnt.clamp(min=1.0).unsqueeze(1)
                per = (torch.sqrt(rm_sse / cnt) * scale).cpu().numpy()
                seen = rm_cnt.cpu().numpy() > 0
                rmse_g = np.where(seen, per.mean(axis=1), np.nan)
            gnames = (list(gait_names) if gait_names is not None
                      else [f"g{i}" for i in range(n_gaits)])
            epoch_s = time.perf_counter() - t_epoch

            # Appended together so an interrupt cannot leave these lists at different lengths.
            hist["train"].append(tr_loss)
            hist["val"].append(va_loss)
            hist["val_sw"].append(vsw)
            hist["gnorm"].append(tr_gnorm)
            hist["sec"].append(epoch_s)
            hist["floor"].append(tr_floor)

            with torch.no_grad():
                ublk = {}
                for b, plist in blocks.items():
                    ublk[b] = float(torch.linalg.vector_norm(torch.stack([
                        torch.linalg.vector_norm(p.detach() - p0)
                        for p, p0 in zip(plist, theta0[b])])))
            # grad_blocks partitions every trainable parameter into one block, so the Euclidean total over
            # blocks is the full update norm.
            upd = math.sqrt(sum(v * v for v in ublk.values()))
            hist["upd"].append(upd)
            gblk = {b: float(bacc[b]) / max(nb, 1) for b in blocks}
            for b in blocks:
                hist[f"g_{b}"].append(gblk[b])
                hist[f"u_{b}"].append(ublk[b])

            flag = ""
            # A real boolean for metrics.csv: `flag` is a display string that also carries the snapshot
            # tag, so parsing `best` from it would log 0 on an epoch that is both.
            is_best = False
            if va_loss < best:
                best = va_loss
                torch.save(model.state_dict(), best_path)
                flag = " *"
                is_best = True

            # Periodic snapshot, saved REGARDLESS of val and independent of best_model.pt: val picks the
            # lowest-scoring epoch, which under --lambda_hold can be an unpenalized one from before any
            # pruning. Gives a fixed trajectory to choose from; load with --ckpt epochN.pt.
            if (args.save_model_every > 0
                    and epoch % args.save_model_every == 0):
                torch.save(model.state_dict(),
                           out_path(out_dir, f"epoch{epoch}.pt"))
                flag += " [saved epoch]"

            if epoch % args.log_every == 0 or epoch == 1:
                print(f"  {epoch:>6}  {tr_loss:>10.6f}  {va_loss:>10.6f}  "
                      f"{vsw:>13.6f}  {opt.param_groups[0]['lr']:>9.2e}"
                      f"  {tr_gnorm:>8.3f}  {upd:>8.3f}  {epoch_s:>6.1f}{flag}")

            # Run the timing diagnostic BEFORE metrics.write so this epoch's rates land in THIS epoch's
            # row (printing is deferred below). Epoch 1 always runs it, so the rate columns exist from
            # the first row, which fixes MetricsWriter's columns.
            run_timing_diag = (timing_diag is not None and
                               (epoch % args.timing_log_every == 0
                                or epoch == 1))
            timing_lines = None
            if run_timing_diag:
                timing_lines, last_timing_stats = timing_diag()

            # One column per (gait, unit) for whichever per-unit series the diagnostic produced, so
            # metrics.csv carries the numbers the printed report does: timing_report 'rate' ->
            # spk_<gait>_t<unit>; drive_report 'fanin90' -> fanin_<gait>_g<sub-network> (driven off which
            # key is present, so the reports stay interchangeable). Names use the gait's NAME, so they
            # survive --gaits reordering. Blank on non-eval epochs (None renders as ""), which separates
            # "not measured" from "measured at zero". dominant_channel is not logged per epoch (it lands
            # in timing_layer_stats at the end).
            spike_rate_cols = {}
            if last_timing_stats:
                for key, prefix, unit in (("rate", "spk", "t"),
                                          ("fanin90", "fanin", "g"),
                                          ("spk_cyc", "dspk", "g"),
                                          ("n_active", "nact", "g"),
                                          ("align_vel", "avel", "g"),
                                          ("align_cor", "acor", "g")):
                    if key not in last_timing_stats[0]:
                        continue
                    for st in last_timing_stats:
                        for u, r in enumerate(st[key]):
                            spike_rate_cols[f"{prefix}_{st['gait']}_{unit}{u}"] = (
                                r if run_timing_diag else None)

            metrics.write({
                "epoch": epoch, "train": tr_loss, "val": va_loss,
                "val_post_switch": vsw, "lr": opt.param_groups[0]["lr"],
                "grad_norm": tr_gnorm, "update_norm": upd,
                "spike_penalty": tr_floor, "sec": epoch_s,
                "best": int(is_best),
                "rmse_mean": (float(np.nanmean(rmse_g))
                              if np.isfinite(rmse_g).any() else None),
                **{f"rmse_{gnames[i]}": (float(rmse_g[i])
                                         if np.isfinite(rmse_g[i]) else None)
                   for i in range(n_gaits)},
                **{f"grad_{b}": gblk[b] for b in sorted(blocks)},
                **{f"upd_{b}": ublk[b] for b in sorted(blocks)},
                **spike_rate_cols,
            })

            # Per-block breakdown on the diagnostic cadence. Read u/g: a block with 1% of the gradient and
            # 60% of the movement is diffusing, not learning.
            if epoch % args.timing_log_every == 0 or epoch == 1:
                gtot = sum(gblk.values()) or 1.0
                utot = sum(ublk.values()) or 1.0
                print(f"      {'block':<10}{'|grad|':>11}{'g%':>7}"
                      f"{'|upd|':>10}{'u%':>7}{'u/g':>8}{'upd/param':>11}")
                for b in sorted(blocks, key=lambda k: -ublk[k]):
                    gs, us = 100*gblk[b]/gtot, 100*ublk[b]/utot
                    npar = sum(p.numel() for p in blocks[b])
                    print(f"      {b:<10}{gblk[b]:>11.3g}{gs:>6.1f}%"
                          f"{ublk[b]:>10.4g}{us:>6.1f}%"
                          f"{(us/gs if gs > 1e-9 else float('inf')):>8.1f}"
                          f"{ublk[b]/math.sqrt(max(npar,1)):>11.2e}")
                # Validation RMSE is not a timing quantity: it prints in this every-N block so dense runs see
                # it too (rmse_g is computed every epoch for every arch and always logged to metrics.csv).
                if np.isfinite(rmse_g).any():
                    print("      val RMSE (deg, mean over joints): "
                          + "  ".join(f"{gnames[i]} {rmse_g[i]:.2f}"
                                      for i in range(n_gaits)
                                      if np.isfinite(rmse_g[i]))
                          + f"   | mean {np.nanmean(rmse_g):.2f}"
                          + "   (validation stream, incl. gait switches)")

            # Timing layer: cheap (timing units only, batch 1) but prints n_gaits lines, so it runs on a
            # slower cadence. The call happened earlier (before metrics.write); this is the print plus
            # the reinit logic.
            if run_timing_diag:
                for ln in timing_lines:
                    print(ln)

                # Failures are tracked per (gait, unit): w_in_gait has a gait axis, so a unit fine for five
                # gaits and dead for the sixth gets only that column repaired. "rate" in the guard because
                # drive_report / drive_spike_report produce no rates, so the rescue does not apply under
                # cpg / cpg_lif (and would KeyError).
                if (args.reinit_dead_after > 0 and last_timing_stats
                        and "rate" in last_timing_stats[0]):
                    n_u = len(last_timing_stats[0]["rate"])
                    pairs = [(gi, u) for gi in range(len(last_timing_stats))
                             for u in range(n_u)]
                    is_dead = {p: last_timing_stats[p[0]]["rate"][p[1]] < 1e-6
                               for p in pairs}
                    is_sat  = {p: (sat_rate is not None and
                                   last_timing_stats[p[0]]["rate"][p[1]]
                                   >= sat_rate)
                               for p in pairs}
                    for p in pairs:
                        dead_streak[p] = (dead_streak.get(p, 0) + 1
                                          if is_dead[p] else 0)
                        sat_streak[p]  = (sat_streak.get(p, 0) + 1
                                          if is_sat[p] else 0)
                    dead_due = [p for p in pairs
                                if dead_streak[p] >= args.reinit_dead_after]
                    sat_due  = [p for p in pairs
                                if sat_streak[p] >= args.reinit_dead_after]
                    if dead_due or sat_due:
                        reinit_timing_units(
                            model, dead_due, sat_due, n_gaits,
                            sat_scale=args.sat_scale)
                        for p in dead_due:
                            dead_streak[p] = 0
                        for p in sat_due:
                            sat_streak[p] = 0

    except KeyboardInterrupt:
        # Return normally rather than propagating, so main() still writes the final config and plots
        # (and ONNX with --export_onnx 1) from the best checkpoint so far.
        done = len(hist["train"])
        print()
        print("  " + "-" * 88)
        print(f"  [INTERRUPT] Ctrl+C received during epoch {done + 1}.")
        print(f"              {done} epoch(s) completed and recorded; the "
              f"partial epoch is discarded.")
        print(f"              Best val MSE so far : {best:.6f}")
        print( "              Stopping training and proceeding to export.")

    metrics.close()
    print("  " + "-" * 88)
    if hist["sec"]:
        tot = sum(hist["sec"])
        print(f"  {len(hist['sec'])} epoch(s) in {tot:.1f}s  "
              f"(mean {tot / len(hist['sec']):.2f}s/epoch, train+val only)")
    return best, hist, last_timing_stats
