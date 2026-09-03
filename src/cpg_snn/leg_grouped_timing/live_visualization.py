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

import time

import numpy as np

from train import (
    GAIT_FILES_BY_N, cfg_get, load_gait_tables, upsample_gait_tables,
)

CPG_C    = "#457b9d"
TIMING_C = "#2a9d8f"
JOINT_C  = "#6a0572"
SNN_C    = "#3d405b"     # decoder blocks: fixed colour, never highlighted
ON_C     = "#e63946"
OFF_A    = 0.60          # edge alpha when idle
DIM      = "#c8cdd4"

# Anatomical names for the k-th joint within a leg, keyed by joints-per-leg.
JOINT_TYPE_NAMES = {2: ["shoulder", "knee"],
                    3: ["coxa", "femur", "tibia"]}


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
        self.type_names = JOINT_TYPE_NAMES.get(
            k, [f"joint type {i}" for i in range(k)])
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

    def update(self, cpg_spk, timing_spk, joints, gait_idx):
        """Cheap on every call; redraws only at the target fps."""
        i = self.n_seen % self.W
        self.buf[i]     = joints
        # Recorded EVERY timestep, unlike the node lighting below which ORs
        # between frames: the plot is the true spike train, the nodes answer
        # "did this neuron fire since the last redraw".
        self.bcpg[i]    = np.asarray(cpg_spk).reshape(-1)[:self.n_cpg]
        self.bgait[i]   = gait_idx
        self.bswitch[i] = False if self.n_seen >= self.W else self.bswitch[i]
        self.n_seen += 1

        # OR spikes into the accumulators so nothing between frames is lost.
        a = np.asarray(cpg_spk).reshape(-1) > 0.5
        self._acc_cpg[:len(a)] |= a
        if self.n_timing:
            b = np.asarray(timing_spk).reshape(-1) > 0.5
            self._acc_tim[:len(b)] |= b

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

    def close(self):
        try:
            self.plt.close(self.fig)
        except Exception:
            pass