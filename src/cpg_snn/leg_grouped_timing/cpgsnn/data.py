"""
Robot and gait data: leg/joint layouts, how joints are partitioned into timing units and
sub-networks, gait-table loading and upsampling, per-step targets, and the smooth-table
derivatives the alignment scores use.
"""

from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d


# ═══════════════════════════════════════════════════════════════════
# Gait-table column layout — EDIT HERE IF YOUR GAIT TABLES DIFFER
# ═══════════════════════════════════════════════════════════════════

# leg l -> (gait-table column for joint A, column for joint B)
LEG_COLS   = [(0, 4), (1, 5), (2, 6), (3, 7)]
N_LEGS     = 4
N_JOINTS   = 8

# Hexapod: 6 legs x 3 joints (coxa, femur, tibia), consecutive per leg; CSV leg order is
# LF, LM, LR, RF, RM, RR.
HEXAPOD_LEG_COLS  = [[3 * l, 3 * l + 1, 3 * l + 2] for l in range(6)]
HEXAPOD_LEG_NAMES = ["LF", "LM", "LR", "RF", "RM", "RR"]
HEXAPOD_N_LEGS    = 6
HEXAPOD_N_JOINTS  = 18


def build_group_cols(n_timing, leg_cols=LEG_COLS, n_joints=N_JOINTS):
    """
    Map sub-network index -> the gait-table output columns it owns.

    `n_timing` is the number of timing neurons, and there is exactly one
    disconnected sub-network per timing neuron, so this also fixes how the
    8 output columns are partitioned:

        n_timing == n_legs   -> group l owns LEG_COLS[l], e.g. (l, l+4)
        n_timing == n_joints -> group j owns column j alone

    Anything else is rejected: a partition into unequal or overlapping
    groups would make `w_out` ragged and silently mis-route servos, so
    there is no third case to guess at.
    """
    n_legs = len(leg_cols)
    if n_timing == n_legs:
        groups = [list(c) for c in leg_cols]
    elif n_timing == n_joints:
        groups = [[j] for j in range(n_joints)]
    else:
        raise ValueError(
            f"n_timing must be n_legs ({n_legs}) or n_joints ({n_joints}), "
            f"got {n_timing}.")

    flat = [c for grp in groups for c in grp]
    if sorted(flat) != list(range(n_joints)):
        raise ValueError(
            f"group columns {groups} are not a partition of "
            f"0..{n_joints - 1} (flattened: {sorted(flat)}).")
    if len({len(g) for g in groups}) != 1:
        raise ValueError(f"groups must be equal size, got {groups}.")
    return groups


STRUCTURE_SHAPES = ("per_leg", "per_joint")


def build_partition(shape, leg_cols, n_joints, what="structure"):
    """
    Output columns owned by each unit of a "per_leg" or "per_joint" structure.

    This is the single primitive behind both --timing_shape and
    --decoder_shape.  Saying a layer is shaped "per_leg" means it has one unit
    per leg and that unit is responsible for that leg's columns; "per_joint"
    means one unit per column.  Expressing both layers in the same currency --
    which output columns does each unit own -- is what lets `build_timing_map`
    connect them without any special-casing per combination.

    `build_group_cols` is the older form of this, keyed on a neuron COUNT
    rather than a name.  It is kept because configs written before
    --timing_shape existed record only `n_timing`, and reconstructing those
    checkpoints has to keep working.
    """
    if shape == "per_leg":
        groups = [list(c) for c in leg_cols]
    elif shape == "per_joint":
        groups = [[j] for j in range(n_joints)]
    else:
        raise ValueError(f"{what} must be one of "
                         f"{'|'.join(STRUCTURE_SHAPES)}, got {shape!r}")

    flat = [c for grp in groups for c in grp]
    if sorted(flat) != list(range(n_joints)):
        raise ValueError(
            f"{what}={shape!r} gives {groups}, not a partition of "
            f"0..{n_joints - 1} (flattened: {sorted(flat)}). Check --leg_cols.")
    if len({len(g) for g in groups}) != 1:
        raise ValueError(f"{what}={shape!r} gives unequal groups {groups}; "
                         f"w1/w_out would be ragged. Check --leg_cols.")
    return groups


def build_timing_map(timing_cols, group_cols):
    """Which timing units feed each sub-network: a (G, K) list of indices.

    The rule: sub-network g takes every timing unit whose output columns overlap g's.
      timing per_joint + decoder per_leg   K=3, disjoint: 18 timing units, 6 decoders,
          each fed by its leg's 3 units (a decoder sees three phase references).
      timing per_leg + decoder per_joint   K=1, SHARED: 6 timing units, 18 decoders;
          a leg's 3 decoders get the same spike train. Legal but nearly pointless.
      matched shapes                       K=1, disjoint (the original model).

    K must be equal across groups because w1 is a dense (G, K, Hg) tensor; it is
    validated because --leg_cols is user-supplied.
    """
    G = len(group_cols)
    tmap = []
    for gcols in group_cols:
        own = set(gcols)
        tmap.append([t for t, tcols in enumerate(timing_cols)
                     if own & set(tcols)])

    if any(not m for m in tmap):
        bad = [g for g, m in enumerate(tmap) if not m]
        raise ValueError(
            f"sub-network(s) {bad} have no timing input: group_cols "
            f"{group_cols} and timing_cols {timing_cols} do not overlap. A "
            f"sub-network with no input is permanently silent.")
    sizes = {len(m) for m in tmap}
    if len(sizes) != 1:
        raise ValueError(
            f"timing fan-in must be equal for every sub-network (w1 is a "
            f"dense (G, K, Hg) tensor); got sizes {sorted(sizes)} from "
            f"group_cols {group_cols} and timing_cols {timing_cols}.")

    used = {t for m in tmap for t in m}
    missing = sorted(set(range(len(timing_cols))) - used)
    if missing:
        raise ValueError(
            f"timing unit(s) {missing} feed no sub-network, so their "
            f"CPG->timing weights would receive zero gradient forever. "
            f"group_cols {group_cols}, timing_cols {timing_cols}.")

    assert len(tmap) == G
    return tmap


# ═══════════════════════════════════════════════════════════════════
# Gait tables
# ═══════════════════════════════════════════════════════════════════
# Loaded from CSV (np.loadtxt on gaits_dir/{name}.csv); the file list is picked from
# --n_cpg_neurons (4 -> quadruped, 6 -> hexapod) unless overridden with --gaits.
QUADRUPED_GAIT_FILES = ["bittle_wkF", "bittle_bk", "bittle_wkL", "bittle_wkR"]
HEXAPOD_GAIT_FILES = [
    "tripod", "tripod_backwards",
    "tripod_right", "tripod_left",
    "ripple", "ripple_backwards",
    "ripple_right", "ripple_left"
]
GAIT_FILES_BY_N = {4: QUADRUPED_GAIT_FILES, 6: HEXAPOD_GAIT_FILES}


def load_gait_tables(names, gaits_dir):
    """Load gaits_dir/{name}.csv for each name (np.loadtxt, comma-delimited, float32).
    Validates equal column counts: upsample_gait_tables interpolates per column, and a
    mismatch there fails far from its cause.
    """
    gaits_dir = Path(gaits_dir)
    tables = []
    for nm in names:
        p = gaits_dir / f"{nm}.csv"
        if not p.exists():
            raise FileNotFoundError(
                f"Gait file not found: {p}\n"
                f"  Looked for {len(names)} file(s) in {gaits_dir.resolve()}: "
                f"{names}\n"
                f"  Pass --gaits_dir to point at the right folder, or "
                f"--gaits to load a different set of names.")
        tables.append(np.loadtxt(p, delimiter=",", dtype=np.float32))

    widths = {t.shape[1] for t in tables}
    if len(widths) > 1:
        detail = "  ".join(f"{nm}={t.shape[1]}" for nm, t in zip(names, tables))
        raise ValueError(
            f"Gait tables have mismatched column counts, cannot form one "
            f"target array: {detail}")
    return tables, list(names)


def default_leg_layout(n_cpg_neurons, n_joints):
    """(n_legs, leg_cols) for a known n_joints, keyed on n_joints alone (leg_cols is
    robot anatomy, not a function of the CPG size). n_joints=8 (quadruped): column j
    and j+4 share a circular phase offset. n_joints=18 (hexapod): leg-major, 3
    columns per leg (coxa, femur, tibia), legs LF/LM/LR/RF/RM/RR. Raises otherwise:
    pass --leg_cols.
    """
    # Keyed on n_joints alone: leg_cols is robot anatomy, which does not change when
    # --n_cpg_neurons goes from 6 to 18. n_cpg_neurons stays in the signature (and the
    # error message) for context.
    if n_joints == 8:
        return N_LEGS, [list(c) for c in LEG_COLS]
    if n_joints == 18:
        return HEXAPOD_N_LEGS, [list(c) for c in HEXAPOD_LEG_COLS]
    raise ValueError(
        f"No known leg layout for n_cpg_neurons={n_cpg_neurons}, "
        f"n_joints={n_joints}. Pass --leg_cols explicitly.")


def upsample_gait_tables(tables, names, target_rows=None, verbose=True):
    """Cubic-interpolate every table to a common row count (equal phase
    resolution across gaits, so per-gait loss isn't skewed by quantiization)."""
    if target_rows is None:
        target_rows = max(t.shape[0] for t in tables)
    out = []
    for t, nm in zip(tables, names):
        if t.shape[0] == target_rows:
            out.append(t.copy())
            if verbose:
                print(f"      {nm:>4s} : {t.shape[0]} rows (unchanged)")
        else:
            x0 = np.linspace(0.0, 1.0, t.shape[0])
            x1 = np.linspace(0.0, 1.0, target_rows)
            f  = interp1d(x0, t, axis=0, kind="cubic", fill_value="extrapolate")
            out.append(f(x1).astype(np.float32))
            if verbose:
                print(f"      {nm:>4s} : {t.shape[0]} -> {target_rows} rows (cubic)")
    return out, int(target_rows)



# ═══════════════════════════════════════════════════════════════════
# Targets
# ═══════════════════════════════════════════════════════════════════

def build_targets(phase, gait_tables, phase_zero=0.0):
    """
    phase       : (T,) float in [0,1), NaN where undefined
    gait_tables : list of (R, 8) upsampled tables

    Returns
    -------
    targets : (G, T, 8) float32, normaliized to [-1, 1] with a single global
              min/max across all gaits (so one denormalization on the robot)
    valid   : (T,) bool
    (lo,hi) : the global min/max used
    """
    T     = len(phase)
    valid = ~np.isnan(phase)
    R     = gait_tables[0].shape[0]

    ph  = np.where(valid, phase, 0.0).astype(np.float64)
    row = (((ph + phase_zero) % 1.0) * R).astype(np.int64) % R

    allv   = np.concatenate([t.ravel() for t in gait_tables])
    lo, hi = float(allv.min()), float(allv.max())

    targets = np.stack([
        ((t[row] - lo) / (hi - lo + 1e-8) * 2.0 - 1.0).astype(np.float32)
        for t in gait_tables])
    targets[:, ~valid] = 0.0
    return targets, valid, (lo, hi)


def joint_demand(table, cols, phase, phase_zero=0.0, order=1):
    """Per-step |d^order theta / dphase^order| for a sub-network's joint column(s), from the
    SMOOTH gait table at each step's exact phase. Not from the per-step target:
    build_targets looks up the lower table row per step, a staircase whose derivatives
    measured 8.4% (1st) and 143% (2nd) error, against 0.1% / 2.1% from the table (512
    rows, 180 steps/cycle). Units are per unit phase (dphase/dstep is constant under the
    fake CPG and the score is chance-anchored, so the conversion would cancel).

    table (R, J): one gait's upsampled table. cols: the sub-network's output columns
    (several are each scaled to their own mean, then RMS-combined, so no joint dominates
    by amplitude). phase (L,): per-step phase, NaN where undefined (demand 0).
    order: 1 = speed, 2 = curvature (the corner signal).
    """
    T = np.asarray(table, dtype=np.float64)[:, list(cols)]
    R = T.shape[0]
    if order == 1:
        d = (np.roll(T, -1, axis=0) - np.roll(T, 1, axis=0)) * (R / 2.0)
    elif order == 2:
        d = (np.roll(T, -1, axis=0) - 2.0 * T
             + np.roll(T, 1, axis=0)) * float(R * R)
    else:
        raise ValueError(f"order must be 1 or 2, got {order!r}")
    d = np.abs(d)
    if d.shape[1] > 1:
        d = d / np.maximum(d.mean(axis=0, keepdims=True), 1e-12)
        d = np.sqrt((d ** 2).mean(axis=1))
    else:
        d = d[:, 0]
    ph = np.asarray(phase, dtype=np.float64)
    ok = np.isfinite(ph)
    x = ((np.where(ok, ph, 0.0) + phase_zero) % 1.0) * R
    i0 = np.floor(x).astype(np.int64) % R
    f = x - np.floor(x)
    out = (1.0 - f) * d[i0] + f * d[(i0 + 1) % R]
    out[~ok] = 0.0
    return out
