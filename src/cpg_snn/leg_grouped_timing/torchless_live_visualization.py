"""
Live visualization of the CPG -> SNN -> joints pipeline
======================================================
A single matplotlib window, updated from inside the inference loop:

  banner    gait-switch announcements: which gaits, and what triggered it.
  schematic CPG neurons -> timing neurons -> joints, as three columns of
            nodes. A node lights up when it spikes, and the edges leaving it
            light up with it, so a spike is visible entering the network and
            arriving at the joints it drives.
  traces    one scrolling plot PER JOINT TYPE (coxa/femur/tibia for a hexapod,
            shoulder/knee for a quadruped), each in [-1, 1]. Normalisation is
            PER GAIT, PER JOINT against that joint's own min and max in the
            active gait table, so a 5-degree joint and a 60-degree joint are
            equally readable. Joint dots in the schematic use the same
            normalisation on a coolwarm map (blue = joint minimum, red = joint
            maximum), so a dot and its trace line always agree.

Both x-axes are sized in CPG CYCLES, not fixed timesteps, so they stay
meaningful whether the period is 352 (real oscillator) or 120 (fake_cpg).

Why matplotlib rather than PyGame/PyQtGraph: it is already a dependency of
visualize_timing.py, so this adds nothing to install, and it draws both the schematic
and the scrolling plots in one figure. Blitting keeps it fast enough.

TIMING CAVEAT: `update` buffers cheaply on every call but only REDRAWS at
--viz_fps, and that redraw happens synchronously in the control loop, costing
~16 ms for an 18-joint hexapod. The inference loop uses absolute deadlines so
phase does not drift permanently -- it bunches and catches up -- but for
anything other than debugging prefer --no_robot or a low --viz_fps.

Needs a display. Over SSH that means X forwarding (ssh -X); headless fails
with a clear message.
"""

import json
import multiprocessing as mp
import os
import queue
import socket
import struct
import threading
import time
from pathlib import Path
from scipy.interpolate import interp1d

import numpy as np

QUADRUPED_GAIT_FILES = ["bittle_wkF", "bittle_bk", "bittle_wkL", "bittle_wkR"]
HEXAPOD_GAIT_FILES = [
    "tripod", "tripod_backwards",
    "tripod_right", "tripod_left",
    "ripple", "ripple_backwards",
    "ripple_right", "ripple_left"
]
GAIT_FILES_BY_N = {4: QUADRUPED_GAIT_FILES, 6: HEXAPOD_GAIT_FILES}

def cfg_get(cfg, key, default=None):
    """
    Look a key up at the config top level, then in `model_detail`, then in the
    verbatim `args` dump.

    Three places because the config records the same fact more than once on
    purpose: the top level is the deployment contract, `model_detail` is the
    reproduction record, and `args` is whatever was actually typed.  Older
    configs (config_version 2, pre-timing-layer) are missing the newer keys
    entirely, hence the default.
    """
    for src in (cfg, cfg.get("model_detail", {}), cfg.get("args", {})):
        if isinstance(src, dict) and src.get(key) is not None:
            return src[key]
    return default

JOINT_TYPE_NAMES = {2: ["shoulder", "knee"],
                    3: ["coxa", "femur", "tibia"]}


def joint_type_names(k):
    """Names for k joints-per-leg, falling back to generic labels."""
    return JOINT_TYPE_NAMES.get(k, [f"joint {i}" for i in range(k)])

def load_gait_tables(names, gaits_dir):
    """
    Load one CSV per name from gaits_dir/{name}.csv.  Identical loader to
    train_snn.py's (np.loadtxt, comma-delimited, float32); the only change
    is that the caller decides the name list instead of it being
    hardcoded, so the same function serves quadruped, hexapod, or a custom
    --gaits list.

    Validates that every table has the same column count: upsample_gait_tables
    interpolates per column, and a width mismatch there fails confusingly far
    from its actual cause.
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

def upsample_gait_tables(tables, names, target_rows=None, verbose=True):
    """Cubic-interpolate every table to a common row count (equal phase
    resolution across gaits, so per-gait loss isn't skewed by quantisation)."""
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

CPG_C    = "#457b9d"
TIMING_C = "#2a9d8f"
JOINT_C  = "#6a0572"
SNN_C    = "#3d405b"     # decoder blocks: fixed colour, never highlighted
ON_C     = "#e63946"
OFF_A    = 0.60          # edge alpha when idle
DIM      = "#c8cdd4"

class LiveVisualizer:
    """
    Call `update()` once per inference timestep and `notify_gait_switch()` on
    a change; everything else is internal.
    """

    def __init__(self, cfg, gaits_dir, trace_cycles=5.0, cpg_cycles=1.5,
                 fps=12.0, backend=None):
        import matplotlib
        # train.py sets Agg at import time (it only ever saves figures), and
        # importing it above means we inherit that -- plt.show() would then
        # silently open nothing. force=True is required because pyplot is
        # already imported by then, so use() has to switch the live backend.
        if backend:
            matplotlib.use(backend, force=True)
        elif matplotlib.get_backend().lower() in ("agg", "template", "pdf",
                                                  "ps", "svg", "cairo"):
            for cand in ("QtAgg", "TkAgg", "Qt5Agg", "GTK4Agg", "GTK3Agg",
                         "MacOSX", "WebAgg"):
                try:
                    matplotlib.use(cand, force=True)
                    break
                except Exception:
                    continue
            else:
                raise SystemExit(
                    "No interactive matplotlib backend available (tried "
                    "QtAgg/TkAgg/GTK/MacOSX/WebAgg), so --viz cannot open a "
                    "window. Install one (e.g. 'pip install pyqt5' or the "
                    "python3-tk system package), pass an explicit backend, or "
                    "drop --viz.")
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection, PatchCollection
        self.plt, self._LC, self._PC = plt, LineCollection, PatchCollection

        self.n_cpg    = int(cfg_get(cfg, "n_cpg_neurons", 4))
        self.n_joints = int(cfg_get(cfg, "n_joints", 8))
        self.names    = list(cfg_get(cfg, "gait_names", []))
        n_timing      = cfg_get(cfg, "n_timing")
        self.n_timing = int(n_timing) if n_timing else 0
        self.groups   = cfg_get(cfg, "group_cols") or []

        # Joint TYPES come from leg_cols (the anatomical layout), not
        # group_cols (the network's grouping) -- with --n_timing 18 every
        # group is a single column, which says nothing about joint type.
        legs = cfg_get(cfg, "leg_cols") or [list(range(self.n_joints))]
        k = len(legs[0])
        self.types = [[leg[i] for leg in legs if i < len(leg)]
                      for i in range(k)]
        self.type_names = joint_type_names(k)
        self.n_legs = len(legs)

        self.ranges = self._joint_ranges(cfg, gaits_dir)

        # Both x-axes are expressed in CPG CYCLES rather than raw timesteps, so
        # they stay meaningful when the period changes (352 for the real
        # oscillator, 120 for fake_cpg). The CPG plot gets a much shorter span
        # than the joint traces -- at 5 cycles the individual spikes in a burst
        # merge into a solid block, and the point of that plot is to see them.
        self.period = max(float(cfg_get(cfg, "cpg_period_steps", 254.0)), 10.0)
        self.trace_cycles, self.cpg_cycles = float(trace_cycles), float(cpg_cycles)

        # ── ring buffers (raw values + the gait they were produced under, so
        # history keeps the normalisation that was true when it was recorded)
        self.W     = int(round(self.trace_cycles * self.period))
        # One ring buffer, sized for the longer window; the CPG plot just reads
        # the most recent W_cpg of it.
        self.W_cpg = min(int(round(self.cpg_cycles * self.period)), self.W)
        self.buf     = np.full((self.W, self.n_joints), np.nan, np.float32)
        self.bcpg    = np.zeros((self.W, self.n_cpg), np.float32)
        self.bgait   = np.zeros(self.W, np.int32)
        self.bswitch = np.zeros(self.W, bool)
        self.n_seen  = 0

        # Spikes are OR-ed together between redraws. The control loop runs at
        # ~60 Hz and drawing at ~12 Hz, so sampling only the current timestep
        # would miss ~4 of every 5 -- a CPG neuron bursting at 50% duty then
        # looked "mostly on" at a random phase, and a sparse timing spike
        # could be missed entirely. Accumulating means a neuron shows lit if
        # it spiked at ANY point since the last frame, which renders a burst
        # as solid rather than flickering.
        self._acc_cpg = np.zeros(self.n_cpg, bool)
        self._acc_tim = np.zeros(max(self.n_timing, 1), bool)

        # One colour per CPG neuron, shared between its node's outline and its
        # row in the spike plot -- that pairing is what ties the two views
        # together, along with matching N0..Nk labels on both.
        cm = self.plt.get_cmap("turbo")
        self.cpg_colors = [cm(0.12 + 0.76 * i / max(self.n_cpg - 1, 1))
                           for i in range(self.n_cpg)]

        self.fps, self._last_draw = float(fps), 0.0
        self._banner_until, self._banner_text = 0.0, ""
        self._need_bg = True
        self._build()

    # ------------------------------------------------------------------
    def _joint_ranges(self, cfg, gaits_dir):
        """
        (n_joints, 2) per-joint (min, max) taken across ALL gaits.

        Deliberately NOT per gait: some gaits have a very small range on some
        joints, and scaling those to full height amplified their noise into
        something unreadable. One scale per joint across every gait keeps a
        low-amplitude gait looking low-amplitude, which is the honest picture,
        and means the traces do not rescale when the gait switches.
        """
        files = cfg_get(cfg, "gait_files")
        if files is None:
            files = GAIT_FILES_BY_N.get(int(cfg_get(cfg, "n_cpg_neurons", 4)))
        tables, _ = load_gait_tables(files, gaits_dir)
        rows = int(cfg_get(cfg, "target_rows",
                           max(t.shape[0] for t in tables)))
        tables, _ = upsample_gait_tables(tables, files, rows, verbose=False)
        allg = np.stack(tables)                      # (n_gaits, rows, n_joints)
        r = np.stack([allg.min((0, 1)), allg.max((0, 1))], axis=-1)
        # A constant joint would divide by zero; give it a unit span so it
        # renders flat at 0 instead of exploding.
        flat = r[:, 1] - r[:, 0] < 1e-6
        r[flat, 1] = r[flat, 0] + 1.0
        return r

    def _norm(self, vals):
        """Raw joint values -> [-1, 1]. Broadcasts over leading axes."""
        lo, hi = self.ranges[:, 0], self.ranges[:, 1]
        return np.clip(2.0 * (vals - lo) / (hi - lo) - 1.0, -1.05, 1.05)

    # ------------------------------------------------------------------
    def _grouped_col(self, x, gap=1.5):
        """(n_joints, 2) node positions, clustered per leg with a gap between."""
        legs = [[c for c in leg] for leg in
                ([list(t) for t in zip(*self.types)] if self.types else [])]
        if not legs or sum(len(l) for l in legs) != self.n_joints:
            n = self.n_joints
            return np.stack([np.full(n, x),
                             np.linspace(0.04, 0.92, n) if n > 1
                             else np.array([0.48])], axis=1)
        slot = np.zeros(self.n_joints)
        y = 0.0
        for leg in legs:
            for c in leg:
                slot[c] = y
                y += 1.0
            y += gap
        span = slot.max() - slot.min()
        ys = (0.04 + 0.88 * (slot - slot.min()) / span if span > 0
              else np.full(self.n_joints, 0.48))
        return np.stack([np.full(self.n_joints, x), ys], axis=1)

    def _build(self):
        plt = self.plt
        K = len(self.types)
        try:
            self.fig = plt.figure(figsize=(18.0, 4.2 + 2.0 * K))
        except Exception as e:
            raise SystemExit(
                f"Could not open a plot window ({e}). A display is required; "
                f"over SSH use 'ssh -X', or drop --viz.")
        # Two columns on the schematic row only: CPG spike history on the left,
        # the network diagram on the right. The banner and the joint traces
        # span both.
        gs = self.fig.add_gridspec(
            2 + K, 2, width_ratios=[1.0, 1.0],
            height_ratios=[0.62, 2.6] + [1.0] * K,
            hspace=0.30, wspace=0.13, top=0.97, bottom=0.06,
            left=0.055, right=0.985)

        # ── banner on its OWN axes ───────────────────────────────────
        # Previously a Text inside ax_net: it extended past ax_net.bbox, and
        # since only that bbox is restored when blitting, the overflow left
        # stale pixels that piled up until the text was illegible.
        self.ax_ban = self.fig.add_subplot(gs[0, :])
        self.ax_ban.axis("off")
        self.ax_ban.set_xlim(0, 1)
        self.ax_ban.set_ylim(0, 1)
        # The badge is a text bbox, so its extent grows with the message. It
        # must fit INSIDE ax_ban: only that bbox is restored when blitting, and
        # a badge overflowing vertically leaves un-restored strips spanning its
        # full width -- which showed as a sliver of red left over from a longer
        # gait name after switching to a shorter one. Hence the taller banner
        # row above and the small pad here.
        self.banner = self.ax_ban.text(
            0.5, 0.5, "", ha="center", va="center", fontsize=15,
            fontweight="bold", color="white", zorder=5,
            bbox=dict(boxstyle="round,pad=0.3", fc=ON_C, ec="none"))
        self.banner.set_visible(False)

        # ── CPG spike history ────────────────────────────────────────
        # One axes, one x-axis, but the y-axis is split into a lane per neuron
        # (row i occupies [i, i+0.78]). Because the CPG bursts in sequence, the
        # active lane steps down the plot over time, so a cycle reads as a
        # diagonal.
        ax = self.ax_cpg = self.fig.add_subplot(gs[1, 0])
        ax.set_xlim(0, self.W_cpg)
        ax.set_ylim(-0.35, self.n_cpg - 0.15)
        ax.set_yticks(np.arange(self.n_cpg))
        ax.set_yticklabels([f"N{i}" for i in range(self.n_cpg)], fontsize=14,
                           fontweight="bold")
        for t, c in zip(ax.get_yticklabels(), self.cpg_colors):
            t.set_color(c)
        ax.set_xlabel("timesteps", fontsize=11)
        ax.set_title(f"CPG spikes  ({self.cpg_cycles:g} cycles)", fontsize=15,
                     fontweight="bold")
        ax.tick_params(axis="x", labelsize=10)
        for i in range(self.n_cpg):
            ax.axhline(i, color="k", lw=0.5, alpha=0.18)
        self.cpg_lines = [
            ax.plot(np.arange(self.W_cpg), np.full(self.W_cpg, np.nan), lw=1.2,
                    color=self.cpg_colors[i], drawstyle="steps-post")[0]
            for i in range(self.n_cpg)]

        # ── schematic ────────────────────────────────────────────────
        ax = self.ax_net = self.fig.add_subplot(gs[1, 1])
        # Columns pulled in from 0/1/2 to make room for the decoder stage and
        # the spike plot without stretching the figure further.
        X_CPG, X_TIM, X_DEC, X_JNT = 0.0, 0.58, 1.12, 1.70
        DEC_W = 0.24
        ax.set_xlim(-0.26, 1.86)
        ax.set_ylim(-0.02, 1.22)
        ax.axis("off")
        col = lambda n, x: np.stack(
            [np.full(n, x), (np.linspace(0.04, 0.92, n) if n > 1
                             else np.array([0.48]))], axis=1)
        self.p_cpg = col(self.n_cpg, X_CPG)
        self.p_tim = col(self.n_timing, X_TIM) if self.n_timing else None
        # Joints clustered by LEG rather than evenly spaced, so the 3 (or 2)
        # joints of one leg read as belonging together. Grouped by leg_cols,
        # not group_cols: with --n_timing 18 every network group is a single
        # column and would give no visual grouping at all.
        self.p_jnt = self._grouped_col(X_JNT)

        # ── "Angle Decoder" boxes, one per timing neuron ─────────────
        # Stands in for that sub-network's hidden layers + readout, which is
        # the honest structure without drawing individual hidden units.
        self.dec_boxes, self.p_dec = [], None
        if self.n_timing:
            span = (self.p_tim[1, 1] - self.p_tim[0, 1]) if self.n_timing > 1 else 0.7
            h = max(min(0.78 * abs(span), 0.14), 0.012)
            self.p_dec = col(self.n_timing, X_DEC)
            for y in self.p_dec[:, 1]:
                self.dec_boxes.append(plt.Rectangle(
                    (X_DEC - DEC_W / 2, y - h / 2), DEC_W, h))
            # Fixed colour, no spike highlight: the block stands for a whole
            # sub-network, so flashing it on its timing spike said less than
            # the edges leaving it already do.
            self.pc_dec = self._PC(self.dec_boxes, edgecolors="k",
                                   linewidths=0.8, zorder=3)
            self.pc_dec.set_facecolor([SNN_C] * self.n_timing)
            ax.add_collection(self.pc_dec)
            ax.text(X_DEC, 1.09, "Angle Decoder", ha="center", va="bottom",
                    fontsize=15, fontweight="bold")
            if self.n_timing <= 8:      # "SNN" is illegible in thinner boxes
                for y in self.p_dec[:, 1]:
                    ax.text(X_DEC, y, "SNN", ha="center", va="center",
                            fontsize=13, fontweight="bold", color="white",
                            zorder=4)
        else:
            self.pc_dec = None

        # Edges as LineCollections: one artist each instead of hundreds of
        # Line2Ds, which matters when blitting every frame.
        if self.n_timing:
            self.e_ct = [(a, b) for a in self.p_cpg for b in self.p_tim]
            # timing -> its decoder, then decoder -> the joints it drives
            self.e_td = [(self.p_tim[i], (X_DEC - DEC_W / 2, self.p_dec[i, 1]))
                         for i in range(self.n_timing)]
            self.e_dj = [((X_DEC + DEC_W / 2, self.p_dec[gi, 1]), self.p_jnt[c])
                         for gi, cols in enumerate(self.groups) for c in cols]
        else:
            self.e_ct = [(a, b) for a in self.p_cpg for b in self.p_jnt]
            self.e_td = self.e_dj = []
        mk = lambda segs, lw: self._LC(
            segs, linewidths=lw, zorder=1,
            colors=[(*self._rgb(DIM), OFF_A)] * len(segs))
        self.lc_ct = mk(self.e_ct, 0.6); ax.add_collection(self.lc_ct)
        self.lc_td = mk(self.e_td, 1.0) if self.e_td else None
        self.lc_dj = mk(self.e_dj, 0.9) if self.e_dj else None
        for lc in (self.lc_td, self.lc_dj):
            if lc is not None:
                ax.add_collection(lc)

        self.s_cpg = ax.scatter(*self.p_cpg.T, s=330, c=[CPG_C] * self.n_cpg,
                                edgecolors=self.cpg_colors, linewidths=2.4,
                                zorder=3)
        self.s_jnt = ax.scatter(*self.p_jnt.T, s=140, c=[JOINT_C] * self.n_joints,
                                edgecolors="k", linewidths=0.6, zorder=3)
        self.s_tim = (ax.scatter(*self.p_tim.T, s=230,
                                 c=[TIMING_C] * self.n_timing, edgecolors="k",
                                 linewidths=0.8, zorder=3)
                      if self.n_timing else None)
        labels = [(X_CPG, f"CPG ({self.n_cpg})"),
                  (X_JNT, f"joints ({self.n_joints})")]
        if self.n_timing:
            labels.append((X_TIM, f"timing ({self.n_timing})"))
        for x, lab in labels:
            ax.text(x, 1.09, lab, ha="center", va="bottom", fontsize=15,
                    fontweight="bold")
        # N0..Nk beside each CPG node, in that neuron's colour: the other half
        # of the pairing with the spike plot's lanes.
        for i, (x, y) in enumerate(self.p_cpg):
            ax.text(x - 0.08, y, f"N{i}", ha="right", va="center", fontsize=13,
                    fontweight="bold", color=self.cpg_colors[i])

        # ── traces: one axes per joint type ──────────────────────────
        cmap = plt.get_cmap("turbo")
        self.tr_axes, self.tr_lines, self.tr_sw = [], [], []
        for ti, cols in enumerate(self.types):
            a = self.fig.add_subplot(
                gs[2 + ti, :], sharex=self.tr_axes[0] if self.tr_axes else None)
            a.set_xlim(0, self.W)
            a.set_ylim(-1.15, 1.15)
            a.set_yticks([-1, 0, 1])
            a.tick_params(labelsize=11)
            a.grid(alpha=0.25)
            a.axhline(0, color="k", lw=0.6, alpha=0.35)
            a.set_ylabel(self.type_names[ti], fontsize=15, fontweight="bold")
            if ti < K - 1:
                a.tick_params(labelbottom=False)
            else:
                a.set_xlabel(f"timesteps, newest at right "
                             f"({self.trace_cycles:g} CPG cycles "
                             f"= {self.W} steps)", fontsize=11)
            lines = []
            for li, c in enumerate(cols):
                ln, = a.plot(np.arange(self.W), np.full(self.W, np.nan), lw=1.3,
                             color=cmap(li / max(self.n_legs - 1, 1)),
                             label=f"leg {li}")
                lines.append((c, ln))
            if ti == 0 and len(cols) > 1:
                a.legend(fontsize=10, ncol=min(len(cols), 6),
                         loc="upper right", framealpha=0.85)
            sw = self._LC([], colors="k", linewidths=1.2, alpha=0.5,
                          linestyles="dashed", zorder=4)
            a.add_collection(sw)
            self.tr_axes.append(a)
            self.tr_lines.append(lines)
            self.tr_sw.append(sw)

        self.fig.canvas.manager.set_window_title("CPG-SNN live")
        # Blitting caches a background bitmap, which a resize invalidates.
        # Rather than locking the window size, re-capture on resize.
        self.fig.canvas.mpl_connect("resize_event",
                                    lambda evt: setattr(self, "_need_bg", True))
        self.plt.show(block=False)

        self._animated = ([self.lc_ct, self.s_cpg, self.s_jnt, self.banner]
                          + ([self.s_tim] if self.s_tim else [])
                          + [lc for lc in (self.lc_td, self.lc_dj)
                             if lc is not None]
                          + self.cpg_lines
                          + [ln for grp in self.tr_lines for _, ln in grp]
                          + list(self.tr_sw))
        for a in self._animated:
            a.set_animated(True)
        self._capture_bg()

    def _capture_bg(self):
        """
        (Re)cache the static background. Called on start and after resize.

        The animated artists must stay animated across this draw: matplotlib
        skips animated artists when drawing, which is exactly what makes the
        result a clean background. Un-animating them first (as this used to)
        BAKED the current traces into the bitmap, so after a resize the frozen
        waveform stayed visible underneath the live one.
        """
        c = self.fig.canvas
        c.draw()
        self.bg_ban = c.copy_from_bbox(self.ax_ban.bbox)
        self.bg_net = c.copy_from_bbox(self.ax_net.bbox)
        self.bg_cpg = c.copy_from_bbox(self.ax_cpg.bbox)
        self.bg_tr  = [c.copy_from_bbox(a.bbox) for a in self.tr_axes]
        self._need_bg = False

    # ------------------------------------------------------------------
    def notify_gait_switch(self, old, new, mode):
        nm = lambda i: self.names[i] if 0 <= i < len(self.names) else str(i)
        self._banner_text = f"[{mode}]   {nm(old)}   \u2192   {nm(new)}"
        self._banner_until = time.perf_counter() + 3.0
        if self.n_seen:
            self.bswitch[(self.n_seen - 1) % self.W] = True

    @staticmethod
    def _fit(a, n):
        """Coerce to exactly n float32 values, zero-padding or truncating."""
        a = np.asarray(a, np.float32).reshape(-1)
        if a.size == n:
            return a
        out = np.zeros(n, np.float32)
        out[:min(a.size, n)] = a[:n]
        return out

    def update(self, cpg_spk, timing_spk, joints, gait_idx):
        """
        Cheap on every call; redraws only at the target fps.

        Inputs are coerced to the expected widths rather than trusted: this is
        fed from a socket in remote mode, and a wrong-length array should
        degrade the plot, not raise inside the render loop.
        """
        joints = self._fit(joints, self.n_joints)
        gait_idx = int(gait_idx) if np.isfinite(gait_idx) else 0
        i = self.n_seen % self.W
        self.buf[i]     = joints
        # Recorded EVERY timestep, unlike the node lighting below which ORs
        # between frames: the plot is the true spike train, the nodes answer
        # "did this neuron fire since the last redraw".
        self.bcpg[i]    = self._fit(cpg_spk, self.n_cpg)
        self.bgait[i]   = gait_idx
        self.bswitch[i] = False if self.n_seen >= self.W else self.bswitch[i]
        self.n_seen += 1

        # OR spikes into the accumulators so nothing between frames is lost.
        self._acc_cpg |= self._fit(cpg_spk, self.n_cpg) > 0.5
        if self.n_timing:
            self._acc_tim |= self._fit(timing_spk, self.n_timing) > 0.5

        now = time.perf_counter()
        if now - self._last_draw < 1.0 / self.fps:
            return
        self._last_draw = now
        self._draw(gait_idx)

    # ------------------------------------------------------------------
    def _draw(self, gait_idx):
        if self._need_bg:
            self._capture_bg()
        c = self.fig.canvas

        on_c, on_t = self._acc_cpg.copy(), self._acc_tim.copy()
        self._acc_cpg[:] = False
        self._acc_tim[:] = False

        # ── banner ───────────────────────────────────────────────────
        c.restore_region(self.bg_ban)
        show = time.perf_counter() < self._banner_until
        self.banner.set_visible(show)
        if show:
            self.banner.set_text(self._banner_text)
        self.ax_ban.draw_artist(self.banner)
        c.blit(self.ax_ban.bbox)

        # ── schematic ────────────────────────────────────────────────
        c.restore_region(self.bg_net)
        self.s_cpg.set_facecolor([ON_C if s else CPG_C for s in on_c])
        if self.s_tim is not None:
            self.s_tim.set_facecolor([ON_C if s else TIMING_C for s in on_t])

        # An edge lights up when its SOURCE node spikes, so a spike is visible
        # propagating CPG -> timing -> joints.
        k = self.n_timing if self.n_timing else self.n_joints
        live = np.repeat(on_c, k)
        self.lc_ct.set_color([(*self._rgb(ON_C), 0.95) if s
                              else (*self._rgb(DIM), OFF_A) for s in live])
        self.lc_ct.set_linewidth([1.8 if s else 0.6 for s in live])
        cur = np.nan_to_num(self._norm(self.buf[(self.n_seen - 1) % self.W]))
        cw = self.plt.get_cmap("coolwarm")
        jnt_c = [cw(0.5 * (v + 1)) for v in cur]
        self.s_jnt.set_facecolor(jnt_c)

        if self.lc_td is not None:
            # timing -> decoder still shows the SPIKE, since that is the event.
            self.lc_td.set_color([(*self._rgb(ON_C), 0.95) if s
                                  else (*self._rgb(DIM), OFF_A)
                                  for s in on_t[:self.n_timing]])
            self.lc_td.set_linewidth([2.2 if s else 1.0
                                      for s in on_t[:self.n_timing]])
            # decoder -> joint lights up with the timing neuron driving that
            # decoder, same as every other edge, so a spike reads as one
            # continuous path CPG -> timing -> decoder -> joints.
            lt = np.array([on_t[gi] for gi, cols in enumerate(self.groups)
                           for _ in cols])
            self.lc_dj.set_color([(*self._rgb(ON_C), 0.95) if s
                                  else (*self._rgb(DIM), OFF_A) for s in lt])
            self.lc_dj.set_linewidth([2.0 if s else 0.9 for s in lt])

        for art in ([self.lc_ct, self.s_cpg, self.s_jnt]
                    + ([self.s_tim] if self.s_tim else [])
                    + [lc for lc in (self.lc_td, self.lc_dj) if lc is not None]):
            self.ax_net.draw_artist(art)
        c.blit(self.ax_net.bbox)

        # ── CPG spike history: a lane per neuron, newest at the right ──
        c.restore_region(self.bg_cpg)
        nc = min(self.n_seen, self.W_cpg)
        idx_c = np.arange(self.n_seen - nc, self.n_seen) % self.W
        xs_c = np.arange(self.W_cpg - nc, self.W_cpg)
        for i, ln in enumerate(self.cpg_lines):
            ln.set_data(xs_c, i + 0.78 * self.bcpg[idx_c, i])
            self.ax_cpg.draw_artist(ln)
        c.blit(self.ax_cpg.bbox)

        # Trace window is longer than the CPG window, so it needs its own slice.
        n = min(self.n_seen, self.W)
        idx = np.arange(self.n_seen - n, self.n_seen) % self.W
        xs = np.arange(self.W - n, self.W)

        # ── traces: roll so newest is at the right edge; x stays fixed ──
        vals = self._norm(self.buf[idx])
        segs = [[(v, -1.15), (v, 1.15)] for v in xs[self.bswitch[idx]]]
        for ti, a in enumerate(self.tr_axes):
            c.restore_region(self.bg_tr[ti])
            for col, ln in self.tr_lines[ti]:
                ln.set_data(xs, vals[:, col])
                a.draw_artist(ln)
            self.tr_sw[ti].set_segments(segs)
            a.draw_artist(self.tr_sw[ti])
            c.blit(a.bbox)

        c.flush_events()

    @staticmethod
    def _rgb(hexstr):
        h = hexstr.lstrip("#")
        return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))

    def pump(self):
        """
        Process GUI events without redrawing.

        Needed when the draw is throttled: without this the window stops
        responding (drags, resizes, the close button) between frames. Uses
        flush_events rather than plt.pause, because pause triggers a
        draw_idle that would fight the cached blit background.
        """
        try:
            self.fig.canvas.flush_events()
        except Exception:
            pass

    def alive(self):
        """False once the window has been closed."""
        return bool(self.plt.fignum_exists(self.fig.number))

    def close(self):
        try:
            self.plt.close(self.fig)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════
# 6.  Out-of-process driver
# ═══════════════════════════════════════════════════════════════════
#
# WHY A PROCESS AND NOT A THREAD. matplotlib's GUI backends are not
# thread-safe, and Tk and Qt both want their event loop on the main thread --
# driving a figure from a worker thread gives intermittent crashes rather than
# a working window. A child process owns matplotlib outright and runs it in
# ITS OWN main thread, which is both correct and simpler.
#
# The control loop's only cost becomes one small array put on a queue, tens of
# microseconds, instead of a ~16 ms redraw. The queue is BOUNDED and the put
# is non-blocking: if the visualiser falls behind, frames are dropped and the
# robot loop carries on. It can never be stalled by the window.

_KIND_FRAME, _KIND_GAIT, _KIND_STOP = 0, 1, 2


def _viz_child(q, cfg, gaits_dir, kwargs):
    """Child entry point: owns the figure, drains the queue, redraws."""
    try:
        viz = LiveVisualizer(cfg, gaits_dir, **kwargs)
    except BaseException as e:                      # includes SystemExit
        print(f"  [viz] could not start: {type(e).__name__}: {e}")
        return
    n_frames = 0
    try:
        while True:
            got = False
            # Drain everything pending before drawing: update() buffers
            # cheaply and only redraws on its own fps schedule, so a backlog
            # collapses into one redraw with all the spikes OR-ed in -- which
            # is exactly the accumulate-between-frames behaviour wanted.
            while True:
                try:
                    kind, payload = q.get_nowait()
                except queue.Empty:
                    break
                got = True
                if kind == _KIND_STOP:
                    viz.close()
                    print(f"  [viz] stopped after {n_frames} frames")
                    return
                if kind == _KIND_GAIT:
                    viz.notify_gait_switch(*payload)
                else:
                    n_cpg, n_tim = viz.n_cpg, viz.n_timing
                    a = payload
                    viz.update(a[1:1 + n_cpg],
                               a[1 + n_cpg:1 + n_cpg + n_tim],
                               a[1 + n_cpg + n_tim:], int(a[0]))
                    n_frames += 1
            if not viz.alive():                     # user closed the window
                print(f"  [viz] window closed after {n_frames} frames")
                return
            if not got:
                # Idle: keep the window responsive and yield the CPU.
                viz.pump()
                time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        viz.close()


class VisualizerProxy:
    """
    Drop-in replacement for LiveVisualizer that runs it in a child process.

    Same three methods the inference loop uses -- update / notify_gait_switch
    / close -- so the caller does not care which one it holds.
    """

    def __init__(self, cfg, gaits_dir, maxsize=256, **kwargs):
        self.n_cpg = int(cfg_get(cfg, "n_cpg_neurons", 4))
        self.n_joints = int(cfg_get(cfg, "n_joints", 8))
        nt = cfg_get(cfg, "n_timing")
        self.n_timing = int(nt) if nt else 0
        self.dropped = self.sent = 0
        # "spawn", not the Linux default "fork": the parent has torch (and
        # possibly a CUDA context) loaded, and forking that is a known hazard.
        # The child re-imports cleanly instead. Costs a few seconds at startup,
        # which is spent during the settle period anyway.
        ctx = mp.get_context("spawn")
        self.q = ctx.Queue(maxsize=maxsize)
        self.proc = ctx.Process(target=_viz_child, daemon=True,
                                args=(self.q, cfg, str(gaits_dir), kwargs))
        self.proc.start()

    def _put(self, kind, payload):
        if self.proc is None or not self.proc.is_alive():
            return
        try:
            self.q.put_nowait((kind, payload))
            self.sent += 1
        except queue.Full:
            # Deliberate: the robot loop is never made to wait on the plot.
            self.dropped += 1

    def update(self, cpg_spk, timing_spk, joints, gait_idx):
        # One contiguous float32 buffer rather than a tuple of arrays, so the
        # whole frame is a single small pickle. Layout: [gait, cpg, timing,
        # joints], unpacked by known lengths in the child.
        self._put(_KIND_FRAME, np.concatenate((
            np.float32([gait_idx]),
            np.asarray(cpg_spk, np.float32).reshape(-1),
            np.asarray(timing_spk, np.float32).reshape(-1),
            np.asarray(joints, np.float32).reshape(-1))))

    def notify_gait_switch(self, old, new, mode):
        self._put(_KIND_GAIT, (int(old), int(new), str(mode)))

    def alive(self):
        return self.proc is not None and self.proc.is_alive()

    def close(self):
        if self.proc is None:
            return
        if self.proc.is_alive():
            self._put(_KIND_STOP, None)
            self.proc.join(timeout=3.0)
            if self.proc.is_alive():
                self.proc.terminate()
        if self.sent or self.dropped:
            pct = 100.0 * self.dropped / max(self.sent + self.dropped, 1)
            print(f"  visualiser: {self.sent} frames sent, {self.dropped} "
                  f"dropped ({pct:.1f}%) — drops mean the plot fell behind, "
                  f"not that the control loop stalled")
        self.proc = None


# ═══════════════════════════════════════════════════════════════════
# 7.  Remote driver (robot -> laptop over a socket)
# ═══════════════════════════════════════════════════════════════════
#
# Stream the DATA, not the pixels. A frame is ~124 bytes, so at 60 Hz this is
# ~7 kB/s. Shipping the rendered figure instead -- X11 forwarding, VNC, a
# remote desktop -- means pushing a 1.8-megapixel framebuffer, which is about
# 35 Mbit/s even at 20:1 compression and roughly 12,000x more traffic. That is
# why those feel laggy over WiFi and this does not: all the drawing happens on
# the laptop, and the robot only says what the numbers were.
#
# The laptop LISTENS and the robot CONNECTS OUT, so the viewer can be up first
# and reconnect independently of run order. Over SSH that is a reverse tunnel:
#
#     laptop$  python live_visualization.py --listen 5555
#     laptop$  ssh -R 5555:localhost:5555 user@robot
#     robot$   python run_inference.py --viz --viz_mode remote
#
# Wire format is 1 byte kind + 4 byte big-endian length + payload. Explicitly
# NOT pickle: a socket should not be handing arbitrary objects to eval-like
# machinery, even down a local tunnel.

_MSG_HELLO = 2          # payload: JSON of the run config
_HDR = struct.Struct(">BI")
# A length field is the one thing a corrupt stream can turn into an enormous
# allocation, so it is bounded. The largest legitimate message is the config
# handshake, a few kB.
_MAX_MSG = 1 << 20


def _send_msg(sock, kind, payload):
    sock.sendall(_HDR.pack(kind, len(payload)) + payload)


def _recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


def _recv_msg(sock):
    """One whole message from a BLOCKING socket. Handshake use only."""
    hdr = _recv_exact(sock, _HDR.size)
    if hdr is None:
        return None
    kind, n = _HDR.unpack(hdr)
    if n > _MAX_MSG:
        raise ValueError(f"message length {n} exceeds the {_MAX_MSG} cap")
    payload = _recv_exact(sock, n) if n else b""
    return None if payload is None else (kind, payload)


class _MsgReader:
    """
    Incremental length-prefixed reader for a NON-BLOCKING socket.

    The point is that partial reads must never lose bytes. Assembling a whole
    message inside a recv loop looks fine until the socket runs dry mid-header
    or mid-payload: the BlockingIOError unwinds, whatever had been read is
    thrown away, and the next read takes its "header" from the middle of a
    frame. From then on every length field is garbage -- which shows up as
    empty payloads and an IndexError on the first field, not as an obvious
    protocol error. So bytes are appended to a persistent buffer and only
    consumed once a complete message is present.
    """

    def __init__(self, sock, chunk=1 << 16):
        self.sock, self.chunk = sock, chunk
        self.buf = bytearray()
        self.closed = False
        self.error = None

    def poll(self):
        """Drain the socket, then return every complete message buffered."""
        while not self.closed:
            try:
                b = self.sock.recv(self.chunk)
            except (BlockingIOError, socket.timeout):
                break
            except OSError as e:
                self.closed, self.error = True, e
                break
            if not b:
                self.closed = True
                break
            self.buf += b

        out = []
        while len(self.buf) >= _HDR.size:
            kind, n = _HDR.unpack_from(self.buf, 0)
            if n > _MAX_MSG:
                self.closed = True
                self.error = ValueError(
                    f"length field {n} over the {_MAX_MSG} cap — stream "
                    f"corrupt, dropping the connection")
                break
            if len(self.buf) < _HDR.size + n:
                break                      # incomplete; wait for more bytes
            out.append((kind, bytes(self.buf[_HDR.size:_HDR.size + n])))
            del self.buf[:_HDR.size + n]
        return out


class VisualizerClient:
    """
    Same surface as VisualizerProxy, but the figure lives on another machine.

    The control loop still only touches a bounded queue. A daemon thread does
    the blocking socket writes, because a network hiccup on the loop's own
    thread would be exactly the stall this is meant to avoid. A thread is fine
    here where it was not for matplotlib: sockets are thread-safe and no GUI
    is involved.
    """

    def __init__(self, cfg, host="localhost", port=5555, maxsize=256,
                 timeout=5.0):
        self.n_cpg = int(cfg_get(cfg, "n_cpg_neurons", 4))
        nt = cfg_get(cfg, "n_timing")
        self.n_timing = int(nt) if nt else 0
        self.sent = self.dropped = 0
        self.q = queue.Queue(maxsize=maxsize)
        self._stop = threading.Event()

        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.settimeout(None)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        _send_msg(self.sock, _MSG_HELLO,
                  json.dumps(_json_safe(cfg)).encode("utf-8"))
        self.thread = threading.Thread(target=self._sender, daemon=True)
        self.thread.start()

    def _sender(self):
        while not self._stop.is_set():
            try:
                kind, payload = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                _send_msg(self.sock, kind, payload)
            except OSError:
                self._stop.set()
                return

    def _put(self, kind, payload):
        if self._stop.is_set():
            return
        try:
            self.q.put_nowait((kind, payload))
            self.sent += 1
        except queue.Full:
            self.dropped += 1

    def update(self, cpg_spk, timing_spk, joints, gait_idx):
        self._put(_KIND_FRAME, np.concatenate((
            np.float32([gait_idx]),
            np.asarray(cpg_spk, np.float32).reshape(-1),
            np.asarray(timing_spk, np.float32).reshape(-1),
            np.asarray(joints, np.float32).reshape(-1))).tobytes())

    def notify_gait_switch(self, old, new, mode):
        m = str(mode).encode("utf-8")
        self._put(_KIND_GAIT, struct.pack(">ii", int(old), int(new)) + m)

    def alive(self):
        return not self._stop.is_set()

    def close(self):
        self._stop.set()
        try:
            self.thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            self.sock.close()
        except Exception:
            pass
        if self.sent or self.dropped:
            pct = 100.0 * self.dropped / max(self.sent + self.dropped, 1)
            print(f"  visualiser: {self.sent} frames sent, {self.dropped} "
                  f"dropped ({pct:.1f}%)")


def _json_safe(o):
    """cfg comes from JSON but may have picked up numpy scalars since."""
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def serve_visualizer(port, gaits_dir, host="", **kwargs):
    """
    Laptop side: wait for the robot to connect, then render locally.

    Loops on accept, so the robot can be restarted without restarting this.
    """
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    print(f"Visualiser listening on {host or '0.0.0.0'}:{port}")
    print(f"Gait tables: {gaits_dir}")
    print("Waiting for the robot to connect "
          "(Ctrl+C to quit) ...")
    try:
        while True:
            conn, addr = srv.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"\nConnected: {addr[0]}:{addr[1]}")
            try:
                _serve_one(conn, gaits_dir, kwargs)
            except Exception as e:
                print(f"  session ended: {type(e).__name__}: {e}")
            finally:
                conn.close()
            print("Waiting for the next connection ...")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        srv.close()


def _serve_one(conn, gaits_dir, kwargs):
    # Handshake on a BLOCKING socket with a deadline, so a client that
    # connects and says nothing cannot wedge the server.
    conn.settimeout(15.0)
    try:
        msg = _recv_msg(conn)
    except (socket.timeout, ValueError, OSError) as e:
        print(f"  no usable handshake ({type(e).__name__}: {e}); dropping")
        return
    if msg is None or msg[0] != _MSG_HELLO:
        print("  expected a config handshake first; dropping")
        return
    try:
        cfg = json.loads(msg[1].decode("utf-8"))
        if not isinstance(cfg, dict):
            raise ValueError("config is not a JSON object")
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as e:
        print(f"  unreadable config ({type(e).__name__}: {e}); dropping")
        return
    print(f"  config received: n_cpg={cfg_get(cfg, 'n_cpg_neurons')} "
          f"n_timing={cfg_get(cfg, 'n_timing')} "
          f"n_joints={cfg_get(cfg, 'n_joints')}")

    viz = LiveVisualizer(cfg, gaits_dir, **kwargs)
    n_cpg, n_tim, n_j = viz.n_cpg, viz.n_timing, viz.n_joints
    # gait + cpg + timing + joints, all float32
    want = 4 * (1 + n_cpg + n_tim + n_j)

    conn.settimeout(0.0)               # non-blocking from here
    reader = _MsgReader(conn)
    n = n_bad = 0
    warned = set()

    def warn(tag, msg):
        # Once per kind of problem: at 60 Hz a per-frame print would itself
        # become the bottleneck.
        if tag not in warned:
            warned.add(tag)
            print(f"  {msg} (reported once)")

    try:
        while True:
            msgs = reader.poll()
            for kind, payload in msgs:
                if kind == _KIND_FRAME:
                    if len(payload) != want:
                        n_bad += 1
                        warn("size", f"frame is {len(payload)} B, expected "
                                     f"{want} B for n_cpg={n_cpg} "
                                     f"n_timing={n_tim} n_joints={n_j} — "
                                     f"skipping mismatched frames")
                        continue
                    a = np.frombuffer(payload, np.float32)
                    viz.update(a[1:1 + n_cpg],
                               a[1 + n_cpg:1 + n_cpg + n_tim],
                               a[1 + n_cpg + n_tim:], int(a[0]))
                    n += 1
                elif kind == _KIND_GAIT:
                    if len(payload) < 8:
                        n_bad += 1
                        warn("gait", "short gait message; skipping")
                        continue
                    old, new = struct.unpack(">ii", payload[:8])
                    viz.notify_gait_switch(
                        old, new, payload[8:].decode("utf-8", "replace"))
                else:
                    n_bad += 1
                    warn("kind", f"unknown message kind {kind}; skipping")

            if reader.closed:
                if reader.error is not None:
                    print(f"  connection dropped: {reader.error}")
                else:
                    print(f"  robot disconnected after {n} frames"
                          + (f" ({n_bad} bad messages skipped)" if n_bad else ""))
                return
            if not viz.alive():
                print(f"  window closed after {n} frames")
                return
            if not msgs:
                viz.pump()
                time.sleep(0.01)
    finally:
        viz.close()


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Render the live CPG-SNN visualisation for a robot running "
                    "run_inference.py --viz_mode remote. Streams ~7 kB/s of "
                    "numbers instead of a framebuffer, so it stays smooth on "
                    "a link where X11 forwarding or VNC will not.")
    ap.add_argument("--listen", type=int, default=5555,
                    help="TCP port to wait for the robot on.")
    ap.add_argument("--bind", type=str, default="",
                    help="Interface to bind. Default all. Leave as-is when "
                         "using an SSH reverse tunnel.")
    ap.add_argument("--gaits_dir", type=str, default="../gaits",
                    help="Local folder of gait CSVs, relative to this file. "
                         "Needed for the per-joint trace scaling.")
    ap.add_argument("--viz_fps", type=float, default=12.0)
    ap.add_argument("--viz_trace_cycles", type=float, default=5.0)
    ap.add_argument("--viz_cpg_cycles", type=float, default=1.5)
    a = ap.parse_args()
    serve_visualizer(
        a.listen, Path(os.path.dirname(os.path.abspath(__file__)), a.gaits_dir),
        host=a.bind, fps=a.viz_fps, trace_cycles=a.viz_trace_cycles,
        cpg_cycles=a.viz_cpg_cycles)


if __name__ == "__main__":
    main()