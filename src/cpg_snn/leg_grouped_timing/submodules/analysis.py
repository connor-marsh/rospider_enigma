"""
Measurements without plotting: drive and timing reports, alignment scores, prediction
replay, spike-train helpers and circular statistics. Shared by the training log and
the visualizers.
"""

import math

import numpy as np
import torch

from .data import joint_demand


# ═══════════════════════════════════════════════════════════════════
# Timing-layer diagnostics
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
@torch.no_grad()
def drive_report(model, n_gaits, gait_names=None, indent="    "):
    """The --drive_source cpg counterpart of `timing_report`, with the same (lines, stats)
    contract so run_training can call any of them. (cpg_lif has `drive_spike_report`.)

    Reports DRIVE SELECTION, not spike rates: under cpg the spikes are the CPG's own at a
    fixed rate, so rates would print a constant; what moves is which channels each
    sub-network listens to, in `s_gate`. Needs no CPG replay: it reads the parameters
    directly, which is sound because s_gate and u sit before the sum over channels (no
    threshold or membrane in between). The quantity is drive_strength: |s_gate| *
    ||u[g,c,:]||_1 under cpg (reached under cpg_lif only through visualize.py's
    --drive_metric gain/product, where it is |s_gate|).

    Per (gait, sub-network):
      dom      the channel supplying the most drive; since channel c bursts at phase
               ~c/n_cpg this is the dominant input PHASE. One that stops moving means
               selection has converged. How well it matches the joint's motion is
               scored by alignment_pair.
      fanin90  channels carrying 90% of the drive: what --drive_lambda is meant to
               reduce (a temporal claim, since channels are phase-localiized).
    """
    # drive_strength, not |s_gate|: under cpg the drive is |s_gate|*||u||_1 (reporting |s_gate|
    # alone made every sub-network look dominated by channel 0: all-ones init, argmax ties
    # go to index 0).
    S = model.drive_strength().detach().float().cpu()        # (n_gaits, G, C)
    C = model.n_neurons

    # Share per sub-network; overall magnitude is discarded.
    share = S / S.sum(dim=2, keepdim=True).clamp(min=1e-12)
    dom      = share.argmax(dim=2)                       # (n_gaits, G)
    domshare = share.max(dim=2).values
    srt      = share.sort(dim=2, descending=True).values
    # +1: the entries strictly under 90% are the channels BEFORE the one that crosses it.
    k90 = (srt.cumsum(dim=2) < 0.90).sum(dim=2) + 1      # (n_gaits, G)

    names = gait_names or [f"g{i}" for i in range(n_gaits)]
    fmt_i = lambda v: " ".join(f"{int(x):>2d}" for x in v)
    lines, stats = [], []
    for gi in range(n_gaits):
        nm = names[gi] if gi < len(names) else f"g{gi}"
        k  = k90[gi]
        lines.append(
            f"{indent}drive {nm:>5s} : dom ch [{fmt_i(dom[gi])}]  "
            f"fanin90 mean {float(k.float().mean()):.1f} "
            f"(min {int(k.min())} max {int(k.max())} of {C})")
        stats.append({
            "gait":              nm,
            "dominant_channel":  [int(v) for v in dom[gi]],
            "dominant_share":    [float(v) for v in domshare[gi]],
            "fanin90":           [int(v) for v in k],
        })
    return lines, stats


@torch.no_grad()
def drive_spike_report(model, spikes, n_gaits, device, period,
                       t0=0, n_steps=None, gait_names=None, indent="    ",
                       align_ctx=None):
    """The --drive_source cpg_lif reporter, same (lines, stats) contract as timing_report
    and drive_report. Reports exact SPIKE COUNTS: with a fixed threshold there is no scale
    ambiguity, and "channel c delivers nothing" is the integer statement "unit (g,c)
    emitted zero spikes", where fanin90 needed an arbitrary 90% cutoff.

    Per (gait, sub-network):
      spk/cyc  spikes per cycle of the OR over the sub-network's C units. Well defined
               because the units are mutually exclusive (one channel each, CPG channels
               never coincide, zero reset), so it is how many input events the
               sub-network receives per cycle; comparable to the timing layer's spk/cyc.
      active   how many of the C channels emitted ANY spike.
      dom ch   the channel contributing the most spikes (dominant input phase).
    """
    L = int(n_steps if n_steps is not None else 6 * period)
    sl = slice(int(t0), int(t0) + L)
    x = torch.as_tensor(spikes[sl], dtype=torch.float32, device=device)
    x = x.unsqueeze(1)                                      # (L, 1, C)
    G, C = model.G, model.n_neurons
    names = gait_names or [f"g{i}" for i in range(n_gaits)]
    n_cyc = max(L / float(period), 1e-9)

    # fanin90 of the GAIN |s_gate|, the quantity the penalty acts on. `active` is exact but
    # coarse (a channel goes silent only below gain ~0.24, so it sits at C for most of
    # training); fanin90 is continuous, the early-warning signal.
    Sg = model.drive_strength().detach().float().cpu()[:n_gaits]  # (ng,G,C)
    sh = Sg / Sg.sum(dim=2, keepdim=True).clamp(min=1e-12)
    srt = sh.sort(dim=2, descending=True).values
    k90 = (srt.cumsum(dim=2) < 0.90).sum(dim=2) + 1                 # (ng,G)

    align_rows = []
    lines, stats = [], []
    for gi in range(n_gaits):
        gt = torch.full((L, 1), gi, dtype=torch.long, device=device)
        spk = model.drive_spikes_only(x, gt)                # (L,1,G,C)
        if spk is None:
            return [f"{indent}(drive_spikes_only unavailable)"], []
        per_ch = spk[:, 0].sum(dim=0).cpu().numpy()         # (G, C) counts
        # Collapsed train: at most one unit per sub-network fires per step, so sum and OR agree
        # (asserted: the collapse rests on it).
        both = spk[:, 0].sum(dim=2)                         # (L, G)
        if float(both.max()) > 1.0 + 1e-6:
            raise AssertionError(
                f"two drive units of one sub-network fired on the same timestep "
                f"(max {float(both.max())}); the collapsed spk/cyc needs zero reset and "
                f"non-overlapping CPG channels.")
        coll = per_ch.sum(axis=1) / n_cyc                   # (G,) spk/cyc
        active = (per_ch > 0).sum(axis=1)                   # (G,)
        dom = per_ch.argmax(axis=1)
        nm = names[gi] if gi < len(names) else f"g{gi}"
        fi = lambda v: " ".join(f"{int(z):>2d}" for z in v)
        ff = lambda v: " ".join(f"{z:>5.1f}" for z in v)
        f90 = k90[gi].numpy()
        # Velocity and corner alignment of each sub-network's collapsed input train, from spikes
        # already in hand.
        av_, ac_ = [float("nan")] * G, [float("nan")] * G
        if align_ctx is not None:
            tr_all = spk[:, 0].cpu().numpy().sum(axis=2)           # (L, G)
            ph_w = np.asarray(align_ctx["phase"][sl], dtype=np.float64)
            for g in range(G):
                av_[g], ac_[g] = alignment_pair(
                    tr_all[:, g], align_ctx["tables"][gi],
                    model.group_cols[g], ph_w, align_ctx["phase_zero"],
                    np.isfinite(ph_w), align_ctx["vel_tau"],
                    align_ctx["cor_tau"])
            align_rows.append((nm, av_, ac_))
        lines.append(f"{indent}drive {nm:>5s} : spk/cyc [{ff(coll)}]")
        lines.append(f"{indent}      {'':>5s}   active  [{fi(active)}] of {C}"
                     f"   dom ch [{fi(dom)}]")
        lines.append(f"{indent}      {'':>5s}   fanin90 [{fi(f90)}] of {C}"
                     f"   (of |s_gate|; mean {f90.mean():.1f})")
        stats.append({
            "gait":     nm,
            "spk_cyc":  [float(v) for v in coll],
            "n_active": [int(v) for v in active],
            # Same key as drive_report, so metrics.csv gets the same fanin_<gait>_g<N> columns.
            "fanin90":  [int(v) for v in f90],
            "dom_ch":   [int(v) for v in dom],
            "align_vel": [float(v) for v in av_],
            "align_cor": [float(v) for v in ac_],
        })
    if align_rows:
        lines += alignment_section(align_rows, indent=indent)
    return lines, stats


def timing_report(model, spikes, phase, period, n_gaits, device,
                  t0, n_steps=1500, gait_names=None, indent="    ",
                  sat_rate=None, align_ctx=None):
    """Per-gait firing statistics for the timing layer: returns formatted lines (and raw
    dicts) for the caller to print every N epochs. Only spk/cyc and the failure warnings
    are printed; phase and R are still computed and returned (the config records them,
    and the cross-gait separation line uses them).

    spk/cyc  spikes per CPG cycle. 0.00 = DEAD: the sub-network gets constant-zero input
             and the w1 slots it feeds get exactly zero gradient, so it does not recover.
             `sat_rate` or above = SATURATED: the unit fires on essentially every CPG
             spike, so its train is the CPG's OR and carries no phase, and it sits where
             the surrogate gradient is ~1e-6. Pass the CEILING (n_cpg * cpg_rate, less a
             little), not _SAT_FRAC of it (a calibration cap; a unit at 0.6 of the
             ceiling still carries phase).
    phase    circular mean of the cycle phase at which the unit fires, in [0,1).
    R        circular concentration in [0,1]; near 0 the spikes are smeared and `phase`
             is meaningless.
    """
    if not hasattr(model, "timing_only"):
        return [], []

    n_steps = int(min(n_steps, len(spikes) - t0))
    if n_steps <= 0:
        return [], []

    x  = torch.as_tensor(spikes[t0:t0 + n_steps], dtype=torch.float32,
                         device=device).unsqueeze(1)          # (L, 1, Nn)
    ph = np.asarray(phase[t0:t0 + n_steps], dtype=np.float64)
    ok = ~np.isnan(ph)
    n_cycles = max(n_steps / float(period), 1e-9)

    lines, stats = [], []
    align_rows = []
    for g in range(n_gaits):
        gg  = torch.full((n_steps, 1), g, dtype=torch.long, device=device)
        spk = model.timing_only(x, gg)[:, 0].cpu().numpy()

        rate, mu, R = [], [], []
        for j in range(spk.shape[1]):
            m = (spk[:, j] > 0.5) & ok
            rate.append(spk[:, j].sum() / n_cycles)
            if m.sum() == 0:
                mu.append(np.nan); R.append(0.0)
                continue
            z = np.mean(np.exp(1j * 2.0 * np.pi * ph[m]))
            mu.append((np.angle(z) / (2.0 * np.pi)) % 1.0)
            R.append(abs(z))

        name = (gait_names[g] if gait_names is not None else f"g{g}")
        fmt  = lambda v, w=5, p=2: " ".join(
            ("  nan" if not np.isfinite(x_) else f"{x_:{w}.{p}f}") for x_ in v)
        lines.append(f"{indent}timing {name:>5s} : spk/cyc [{fmt(rate)}]")
        dead = [j for j, r in enumerate(rate) if r < 1e-6]
        if dead:
            lines.append(f"{indent}       {'':>5s}   "
                         f"WARNING: timing neuron(s) {dead} DEAD for "
                         f"{name} — sub-network(s) they feed get no input.")
        if sat_rate:
            sat = [j for j, r in enumerate(rate) if r >= sat_rate]
            if sat:
                lines.append(f"{indent}       {'':>5s}   "
                             f"WARNING: timing neuron(s) {sat} SATURATED for "
                             f"{name} (>= {sat_rate:.0f} spk/cyc) — firing on "
                             f"most CPG spikes, so they carry no phase.")
        # Velocity and corner alignment per SUB-NETWORK, from the timing units feeding it
        # (timing_map).
        av_, ac_ = [], []
        if align_ctx is not None and hasattr(model, "timing_map_list"):
            ph_w = np.asarray(ph, dtype=np.float64)
            for gs, units in enumerate(model.timing_map_list):
                a1, a2 = alignment_pair(
                    spk[:, list(units)].sum(axis=1), align_ctx["tables"][g],
                    model.group_cols[gs], ph_w, align_ctx["phase_zero"],
                    np.isfinite(ph_w), align_ctx["vel_tau"],
                    align_ctx["cor_tau"])
                av_.append(a1); ac_.append(a2)
            align_rows.append((name, av_, ac_))
        stats.append({"gait": name, "rate": [float(v) for v in rate],
                      "phase": [None if not np.isfinite(v) else float(v)
                                for v in mu],
                      "R": [float(v) for v in R],
                      "align_vel": [float(v) for v in av_],
                      "align_cor": [float(v) for v in ac_]})

    # THE headline diagnostic for gait separation: near-identical phase vectors across gaits mean
    # the timing layer collapsed to a gait-independent clock and the sub-networks carry all
    # the gait knowledge. Identical rate and R across gaits is EXPECTED (the spike penalty
    # pulls every pair to the CPG's values), so phase is the only statistic left to differ.
    if len(stats) > 1:
        ph = np.array([[np.nan if v is None else v for v in st["phase"]]
                       for st in stats], dtype=np.float64)
        with np.errstate(invalid="ignore"):
            z = np.exp(1j * 2.0 * np.pi * ph)
            spread = np.nanmax(np.abs(z - np.nanmean(z, axis=0, keepdims=True)))
        if np.isfinite(spread):
            lines.append(f"{indent}phase separation across gaits: "
                         f"{spread:.3f}  (max deviation on the unit circle; "
                         f"below ~0.1 means the gaits are NOT being "
                         f"distinguished)")
    if align_rows:
        lines += alignment_section(align_rows, indent=indent)
    return lines, stats


# Alignment credit horizons, in steps. FIXED (not derived from the learned taus) so the
# yardstick is the same across runs, readout-tau experiments included. velocity: ~a drive
# spike's propagation delay through the two hidden layers. corner: ~one step, i.e. credit
# for being AT a corner, not for anticipating it (the cubic upsampling already spreads
# |theta''| over a few steps).
_ALIGN_VEL_TAU = 3.0
_ALIGN_COR_TAU = 1.0


def alignment_score(train, demand, tau, valid=None):
    """How well a sub-network's input spikes coincide with a per-step demand signal,
    chance-anchored so it is comparable across joints. Used with demand = |speed|
    (velocity score) and |curvature| (corner score), both from joint_demand.

    Per-step credit is a DECAYED MAXIMUM of the demand v:
        w(t) = max over j >= 0 of  r^j * v(t + j),     r = exp(-1/tau)
    so a spike gets the demand at its own step in full, partial credit for higher demand
    shortly AHEAD, and none for demand behind it (which it cannot have caused). (A decayed
    sum placed the best spike before fast motion under --synaptic.)
        S = (spike-weighted mean of w - mean of w over all steps) / (max of w - mean of w)
    S = 0 is random placement, 1 = every spike at peak demand, < 0 = spikes where demand
    is low (unbounded below and waveform-dependent, so compare only positive scores
    across joints). NaN if there are no spikes. Anchored to the MEAN: under max-only
    normaliization random placement scores ~0.64 on a sinusoid but ~0.25 on a sharp one.

    train (L,) non-negative spike weights; demand (L,) from joint_demand; tau credit
    horizon in steps; valid (L,) bool. The window must be a whole number of cycles of ONE
    gait so the circular wrap lands on the same phase.
    """
    v = np.asarray(demand, dtype=np.float64)
    L = v.shape[0]
    r = math.exp(-1.0 / max(float(tau), 1e-6))
    # w(t) = max(v(t), r * w(t+1)), wrapped circularly; repeated passes
    # converge because r < 1.
    w = v.copy()
    for _ in range(200):
        prev = w.copy()
        nxt = w[0]
        for t in range(L - 1, -1, -1):
            cand = r * nxt
            if cand > w[t]:
                w[t] = cand
            nxt = w[t]
        if np.array_equal(w, prev):
            break
    ok = (np.ones(L, dtype=bool) if valid is None
          else np.asarray(valid, dtype=bool)[:L])
    s = np.asarray(train, dtype=np.float64)[:L] * ok
    if s.sum() <= 0 or not ok.any():
        return float("nan")
    base, top = float(w[ok].mean()), float(w[ok].max())
    if top - base <= 1e-12:
        return float("nan")
    return float(((s * w).sum() / s.sum() - base) / (top - base))


def alignment_pair(train, table, cols, phase, phase_zero, valid=None,
                   vel_tau=_ALIGN_VEL_TAU, cor_tau=_ALIGN_COR_TAU):
    """(velocity score, corner score) for one sub-network over one window."""
    return (alignment_score(train, joint_demand(table, cols, phase,
                                                phase_zero, 1),
                            vel_tau, valid),
            alignment_score(train, joint_demand(table, cols, phase,
                                                phase_zero, 2),
                            cor_tau, valid))


def alignment_section(per_gait, indent="      "):
    """The printed ALIGNMENT block: one line per gait, grouped by statistic with the two
    scores side by side (max, min, mean, each as (vel, cor)).
    per_gait: list of (gait_name, vel_scores, cor_scores).
    """
    def st(a):
        a = np.asarray(a, dtype=np.float64)
        f = a[np.isfinite(a)]
        if f.size == 0:
            return (float("nan"),) * 3
        return float(f.max()), float(f.min()), float(f.mean())
    fmt = lambda x: "  n/a" if not np.isfinite(x) else f"{x:+.2f}"
    out = [f"{indent}alignment   (velocity, corner)   "
           f"0 = random, 1 = ideal"]
    for nm, vel, cor in per_gait:
        mv, nv, av = st(vel)
        mc, nc, ac = st(cor)
        dead = int((~np.isfinite(np.asarray(vel, dtype=np.float64))).sum())
        out.append(f"{indent}  {nm:>10s} : max ({fmt(mv)}, {fmt(mc)})   "
                   f"min ({fmt(nv)}, {fmt(nc)})   "
                   f"mean ({fmt(av)}, {fmt(ac)})"
                   + (f"   [{dead} with no input]" if dead else ""))
    return out


# ═══════════════════════════════════════════════════════════════════
# Model probes
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def timing_raster(model, spikes, gait_idx, device):
    """(T, n_timing) exact timing-layer spikes for a constant gait."""
    x  = torch.as_tensor(spikes, dtype=torch.float32, device=device).unsqueeze(1)
    gg = torch.full((x.shape[0], 1), int(gait_idx),
                    dtype=torch.long, device=device)
    return model.timing_only(x, gg)[:, 0].cpu().numpy()


@torch.no_grad()
def predict_and_membranes(model, spikes, gait_idx, device, warm, tgt_range,
                          keep_membranes=True, max_mem_steps=1500):
    """Free-run the full model; returns predictions in DEGREES for steps [warm:] plus
    per-timestep membrane snapshots if asked. Steps model.step manually because forward()
    discards the intermediate state the membranes need.
    """
    lo, hi = tgt_range
    scale, shift = (hi - lo) / 2.0, (hi + lo) / 2.0

    x = torch.as_tensor(spikes, dtype=torch.float32, device=device)
    T = x.shape[0]
    # Constant across the run, so built once rather than inside the T-step loop.
    g = torch.full((1,), int(gait_idx), dtype=torch.long, device=device)
    state = model.init_state(1, device)
    preds, mems = [], []
    for t in range(T):
        y, state, _ = model.step(x[t:t + 1], g, state)
        preds.append(y[0].cpu().numpy())
        # Capped: a state snapshot is ~12 KB, so a long window would cost hundreds of MB for traces the
        # membrane plot never reads.
        if keep_membranes and warm <= t < warm + max_mem_steps:
            # state: 9 tensors for timing_grouped (see TimingGroupedSNN), (mem1, mem2, memo) for dense.
            mems.append([s[0].cpu().numpy().copy() for s in state])

    pred = np.stack(preds)[warm:] * scale + shift
    return pred, mems


def gt_degrees(gait_tables, phase, gait_idx, phase_zero):
    """Gait-table angles at each timestep's phase, in degrees.  Mirrors
    build_targets' row indexing exactly (same modulo, same rounding)."""
    tbl = gait_tables[gait_idx]
    R   = tbl.shape[0]
    ph  = np.where(np.isnan(phase), 0.0, phase)
    row = (((ph + phase_zero) % 1.0) * R).astype(np.int64) % R
    return tbl[row]


# ═══════════════════════════════════════════════════════════════════
# Phase statistics
# ═══════════════════════════════════════════════════════════════════

def drive_channels(share, frac=0.90, alpha_min=0.38):
    # frac=None selects every STRICTLY POSITIVE channel: right once the drive is a spike COUNT (cpg_lif),
    # where "zero spikes" is exact and a 90% rule would hide channels that really fire (it keeps 15 of
    # 18 at a near-uniform row). The 90% rule stays correct for the continuous drive under cpg, where
    # nothing is exactly zero and some cutoff is unavoidable.
    """Which channels to draw for one sub-network, and how strongly. `share` is that
    sub-network's normaliized drive over the C CPG channels; returns (idx, alpha), idx
    sorted by descending share.

    Each obvious approach fails alone: all channels with alpha proportional to share is
    unreadable at 18 channels (a flat row is 18 faint rugs in a gray smear), and a top-k
    loses the gradation and needs an arbitrary cutoff. So keep the channels carrying `frac`
    of the drive (the same 90% rule as the fan-in metric, so plot and number agree) and set
    alpha by share RELATIVE TO THE LARGEST: the dominant channel is always opaque, so a
    concentrated sub-network reads as one solid rug and a flat one as many mid-alpha rugs.
    """
    share = np.asarray(share, dtype=np.float64)
    tot = share.sum()
    if not np.isfinite(tot) or tot <= 0:
        return np.empty(0, int), np.empty(0, float)
    p = share / tot
    order = np.argsort(-p)
    if frac is None:
        idx = order[p[order] > 0.0]
        if idx.size == 0:
            return np.empty(0, int), np.empty(0, float)
    else:
        keep = int((np.cumsum(p[order]) < frac).sum()) + 1
        idx = order[:keep]
    # Raw share-relative alpha, mapped into [alpha_min, 1]: without the floor the weakest member of the
    # 90% set gets ~0.03 (drawn but invisible), so rugs could not be counted against the fan-in. The
    # floor says "faint", not "absent".
    rel = p[idx] / p[idx].max()
    alpha = alpha_min + (1.0 - alpha_min) * rel
    return idx, alpha


def circular_stats(phases, weights=None):
    """Circular mean in [0,1) and concentration R in [0,1] of a set of
    cycle phases.  R is the thing to read: a unit can fire at a healthy rate
    and still be useless if R is low, because then it is firing all over the
    cycle and its mean phase means nothing.

    `weights` lets a spike count for less than one. Used by --drive_source
    cpg / cpg_lif, where a sub-network's effective input phase is the
    DRIVE-WEIGHTED mean over the channels feeding it: a channel supplying 40%
    of the drive should move that mean twice as far as one supplying 20%."""
    p = np.asarray(phases, dtype=np.float64)
    w = (np.ones_like(p) if weights is None
         else np.asarray(weights, dtype=np.float64))
    ok = np.isfinite(p) & np.isfinite(w)
    p, w = p[ok], w[ok]
    if p.size == 0 or w.sum() <= 0:
        return float("nan"), 0.0
    z = np.sum(w * np.exp(1j * 2.0 * np.pi * p)) / w.sum()
    return float((np.angle(z) / (2.0 * np.pi)) % 1.0), float(abs(z))


def fundamental_phase(cycle_traj):
    """Phase of the first Fourier component of one cycle of a waveform, in [0,1): the
    reference for "where in its cycle is this leg". Chosen over "swing onset", which needs an
    arbitrary threshold or derivative test that behaves differently across gaits; the
    fundamental is parameter-free, robust to noise and comparable across gaits. It is a
    consistent reference, not a biomechanical event (for a strongly non-sinusoidal
    trajectory it differs from the visible footfall instant).
    """
    y = np.asarray(cycle_traj, dtype=np.float64)
    y = y - y.mean()
    if not np.any(np.abs(y) > 1e-12):
        return float("nan")
    F = np.fft.rfft(y)
    if len(F) < 2:
        return float("nan")
    return float((np.angle(F[1]) / (2.0 * np.pi)) % 1.0)


def circ_diff(a, b):
    """Signed circular difference a - b, wrapped to [-0.5, 0.5) cycles.
    The exact antipode lands at -0.5 rather than +0.5; the magnitude is what
    the residual is read for, so the sign convention there is immaterial."""
    if not (np.isfinite(a) and np.isfinite(b)):
        return float("nan")
    return float((a - b + 0.5) % 1.0 - 0.5)


def unit_train(tspk, g, ch):
    """One unit's binary train for either raster layout: (L, lanes) where a lane IS the
    unit (timing layer, raw CPG channels), or (L, G, C) under cpg_lif, where the unit is
    (sub-network, channel) because the drive LIF thins and delays its channel's burst, so
    plotting the raw channel would show input the sub-network never received.
    """
    return tspk[:, g, ch] if tspk.ndim == 3 else tspk[:, ch]
