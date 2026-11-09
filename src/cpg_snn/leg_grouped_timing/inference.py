"""
CPG -> SNN inference on the robot  (PyTorch)
============================================
Loads best_model.pt + cpg_lif_snn_config.json from a training run, replays the
CPG, steps the network one timestep at a time, and publishes joint commands to
the servo controller.

Everything about the model comes from the config, through the same
`build_model_from_cfg` / `load_run` that visualize.py uses -- so architecture,
gate mode, taus, group layout and gait set are whatever that checkpoint was
trained with, including for configs predating any given option.  No ONNX.

The one thing NOT in the config is the servo wiring: gait-table column ->
servo id, and the sign/offset conventions of the physical robot.  Training has
no business knowing those, so they live here (--servo_ids).

Usage
-----
    python run_inference.py --model_dir test1
    python run_inference.py --model_dir test1 --no_robot     # offline check
    python run_inference.py --model_dir test1 --gait ripple
    python run_inference.py --model_dir test1 --cycle_time 1.5
    python run_inference.py --model_dir test1 --gait_schedule 0:0,8:1,16:0
"""

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

from submodules.common import cfg_get, outputs_path
from submodules.cpg import LIFCPGStepper
from submodules.run_io import load_run

# Hexapod servo ids, in gait-table column order (18 columns -> 18 servos).
# Taken from the previous ONNX inference script.
HEXAPOD_SERVO_IDS = [5, 3, 1, 11, 9, 7, 17, 15, 13,
                     18, 16, 14, 12, 10, 8, 6, 4, 2]


# ═══════════════════════════════════════════════════════════════════
# 1.  CPG source (one timestep at a time)
# ═══════════════════════════════════════════════════════════════════

class CPGSource:
    """
    Yields one (n_cpg,) spike vector per call, matching what training used.

    Real CPG: stepped live -- it is cheap (numpy, N<=6) and runs indefinitely.
    Fake CPG: fake_step_chunk is a block generator, not a stepper, so one
    period is precomputed and cycled.  Exact, since the pattern is periodic by
    construction.
    """

    def __init__(self, cfg):
        c = dict(cfg.get("cpg", {}))
        self.N = int(c.get("N") or cfg_get(cfg, "n_cpg_neurons", 4))
        W = np.asarray(c["W"], dtype=np.float64) if "W" in c else None
        self.cpg = LIFCPGStepper(
            N=self.N, W=W,
            i_app          = float(c.get("i_app", 8.0)),
            vth_main       = float(c.get("vth_main", 100.0)),
            du_main        = float(c.get("du_main", 0.1)),
            dv_main        = float(c.get("dv_main", 0.3)),
            refrac_main    = int(c.get("refrac_main", 1)),
            vth_fb         = float(c.get("vth_fb", 100.0)),
            du_fb          = float(c.get("du_fb", 1.0)),
            dv_fb          = float(c.get("dv_fb", 0.0)),
            refrac_fb      = int(c.get("refrac_fb", 1)),
            from_fb_weight = float(c.get("from_fb_weight", -1e6)),
            to_fb_weight   = float(c.get("to_fb_weight", 10.0)))

        self.fake = bool(cfg_get(cfg, "fake_cpg", False))
        if self.fake:
            # One period, then cycle. fake_step_chunk's period is
            # n_spikes * N where n_spikes = (vth_fb//to_fb)*(refrac+1).
            n_spikes = int((self.cpg.vth_fb // self.cpg.to_fb_weight)
                           * (self.cpg.refrac_main + 1))
            self.loop = self.cpg.fake_step_chunk(n_spikes * self.N)
            self.i = 0
        else:
            self.cpg.step_chunk(int(c.get("warmup", 2000)))

    def step(self):
        if self.fake:
            row = self.loop[self.i % len(self.loop)]
            self.i += 1
            return row
        return self.cpg.step()


# ═══════════════════════════════════════════════════════════════════
# 2.  Gait selection
# ═══════════════════════════════════════════════════════════════════
#
# The inference loop only ever READS `GaitState.index`.  Every way of changing
# the gait is a GaitSelector subclass that writes to that one shared state from
# its own thread, so the loop carries no selection logic and several selectors
# can run at once (e.g. keyboard + gesture).  If two write close together, the
# later write wins.  The intended end state is one ROS node per method; each
# would subclass GaitSelector the same way and drop in without touching the loop.

# W/A/S/D -> gait-name suffix.  The names come from the gait CSVs themselves
# (tripod, tripod_left, tripod_backwards, tripod_right, ...), so a base name
# plus a suffix is all that is needed -- nothing here is hardcoded to tripod.
WASD_SUFFIX = {"w": "", "a": "_left", "s": "_backwards", "d": "_right"}
# Gestures either pick the direction (keeping the current base gait) or the
# base gait (keeping the current direction where that gait has it).
GESTURE_SUFFIX = {"one": "", "four": "_left", "three": "_backwards", "two": "_right"}
GESTURE_BASE = {"fist": "stand", "gun": "tripod", "five": "ripple"}


def split_gait(name):
    """'ripple_left' -> ('ripple', '_left');  'tripod' -> ('tripod', '')."""
    for suffix in sorted(set(GESTURE_SUFFIX.values()), key=len, reverse=True):
        if suffix and name.endswith(suffix):
            return name[:-len(suffix)], suffix
    return name, ""


class GaitState:
    """Thread-safe current gait index, shared by every selector."""

    def __init__(self, names, initial=0):
        self.names = list(names)
        self._idx  = int(initial)
        self._lock = threading.Lock()
        # Set by the caller; receives (old_index, new_index, kind) so a
        # listener (e.g. the live visualizer) can report both the transition
        # and what triggered it, without the selectors knowing about it.
        self.on_change = None

    @property
    def index(self):
        with self._lock:
            return self._idx

    def set_index(self, i, kind="", why=""):
        if not 0 <= i < len(self.names):
            print(f"  ignoring gait index {i} (valid 0..{len(self.names)-1})")
            return
        with self._lock:
            old, changed = self._idx, i != self._idx
            self._idx = i
        if changed:
            print(f"  gait -> {self.names[i]} [{i}]{why}")
            if self.on_change is not None:
                self.on_change(old, i, kind)


class GaitSelector:
    """One source of gait changes, writing to a shared GaitState."""

    kind = "none"

    def __init__(self, state):
        self.state   = state
        self.names   = state.names
        self._stop   = threading.Event()
        self._thread = None

    @property
    def index(self):
        return self.state.index

    def set_index(self, i, why=""):
        self.state.set_index(i, self.kind, why)

    def describe(self):
        return self.kind

    def start(self):
        """Called once, after settling. Subclasses spawn their listener here."""

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _spawn(self, target):
        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()


class ScheduledGait(GaitSelector):
    """Switch on a wall-clock schedule derived from cycle counts."""

    kind = "schedule"

    def __init__(self, state, schedule, cycle_time):
        super().__init__(state)
        self.schedule, self.cycle_time = schedule, cycle_time

    def describe(self):
        return f"{self.kind}: " + "  ".join(
            f"{c:g}cyc->{self.names[i]}" for c, i in self.schedule)

    def start(self):
        self._spawn(self._loop)

    def _loop(self):
        # Absolute deadlines from a single t0, so a slow set_index cannot make
        # later switches drift.
        t0 = time.perf_counter()
        for cyc, idx in self.schedule:
            wait = t0 + cyc * self.cycle_time - time.perf_counter()
            if wait > 0 and self._stop.wait(wait):
                return
            self.set_index(idx, f"  [schedule @ {cyc:g} cyc]")


class KeyboardGait(GaitSelector):
    """WASD on stdin. W=forward, A=left, S=backwards, D=right."""

    kind = "keyboard"

    def __init__(self, state, base):
        super().__init__(state)
        self.base = base
        self.targets, missing = {}, []
        for key, suffix in WASD_SUFFIX.items():
            name = base + suffix
            if name in self.names:
                self.targets[key] = self.names.index(name)
            else:
                missing.append((key.upper(), name))
        if not self.targets:
            raise SystemExit(
                f"--base_gait {base!r} yields no usable names. Config has: "
                f"{self.names}")
        if missing:
            print(f"  no gait for: " + ", ".join(f"{k}={n}" for k, n in missing)
                  + " (those keys will do nothing)")

    def describe(self):
        return f"{self.kind}: " + "  ".join(
            f"{k.upper()}={self.names[i]}" for k, i in self.targets.items())

    def start(self):
        if not sys.stdin.isatty():
            print("  stdin is not a TTY -- keyboard control disabled, gait "
                  "stays fixed")
            return
        print(f"  keys: " + "  ".join(f"{k.upper()}={self.names[i]}"
                                     for k, i in self.targets.items()))
        self._spawn(self._loop)

    def _loop(self):
        # cbreak rather than raw: it leaves ISIG enabled so Ctrl+C still
        # reaches the main thread. Settings are restored in the finally.
        import select
        import termios
        import tty
        fd = sys.stdin.fileno()
        saved = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not self._stop.is_set():
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    ch = sys.stdin.read(1).lower()
                    if ch in self.targets:
                        self.set_index(self.targets[ch], f"  [key {ch.upper()}]")
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)


# Gesture classification, from Hiwonder's hand_gesture.py demo on the robot.
# Each finger's bend is the angle between two landmark vectors (MediaPipe hand
# landmark indices; 0 = wrist, 4/8/12/16/20 = finger tips).  The demo took
# abs() of a signed angle, which is exactly the unsigned angle used here.
_ANGLE_A = ([3, 0, 0, 0, 0], [4, 6, 10, 14, 18])     # thumb, index, ..., pinky
_ANGLE_B = ([0, 7, 11, 15, 19], [2, 8, 12, 16, 20])

# B = bent, S = straight; the thumb uses its own bent threshold, and "one" only
# needs the thumb above 5 degrees.  First match wins, in the demo's order.
_GESTURES = (("fist", "BBBBB"), ("gun", "SSBBB"), ("hand_heart", "SBBBB"),
             ("one", "xSBBB"), ("two", "BSSBB"), ("three", "BSSSB"),
             ("OK", "BBSSS"), ("four", "BSSSS"), ("five", "SSSSS"),
             ("six", "SBBBS"))


def finger_angles(pts):
    """(21, 2) landmarks in pixels -> (5,) bend angles in degrees."""
    a = pts[_ANGLE_A[0]] - pts[_ANGLE_A[1]]
    b = pts[_ANGLE_B[0]] - pts[_ANGLE_B[1]]
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    return np.degrees(np.arctan2(np.abs(cross), (a * b).sum(1)))


def classify_gesture(angles, bent=65.0, bent_thumb=53.0, straight=49.0):
    """Finger angles (thumb..pinky, degrees) -> gesture name, or "none"."""
    ok = {"B": lambda a, i: a > (bent_thumb if i == 0 else bent),
          "S": lambda a, i: a < straight,
          "x": lambda a, i: a > 5.0}
    for name, pattern in _GESTURES:
        if all(ok[c](a, i) for i, (c, a) in enumerate(zip(pattern, angles))):
            return name
    return "none"


class GestureGait(GaitSelector):
    """
    Hand gestures from the robot camera, via MediaPipe. Nothing is displayed.
    Direction (keeps the current base gait):
        one = forward, two = right, three = backwards, four = left
    Base gait (keeps the current direction if that gait has it, else plain):
        fist = stand, gun = tripod, five = ripple
    A gesture acts once it is seen on `confirm_frames` consecutive frames, and
    not again until a different gesture is confirmed or the hand leaves view,
    so a held hand does not keep overriding the other selectors.
    """

    kind = "gesture"

    def __init__(self, state, topic, confirm_frames=5):
        super().__init__(state)
        self.topic, self.confirm = topic, int(confirm_frames)
        # Imported here rather than in the thread, so a missing package fails
        # at start-up instead of silently leaving gestures dead.
        import mediapipe as mp
        import rclpy
        from cv_bridge import CvBridge
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from sensor_msgs.msg import Image  # type: ignore
        self._mp, self._rclpy, self._Node = mp, rclpy, Node
        self._Executor, self._Image, self._bridge = (
            SingleThreadedExecutor, Image, CvBridge())
        self._last, self._run, self._fired = None, 0, None
        bases = {split_gait(n)[0] for n in self.names}
        missing = [f"{g}={b}" for g, b in GESTURE_BASE.items() if b not in bases]
        if missing:
            print(f"  no gait for: {', '.join(missing)} (those gestures will "
                  f"do nothing)")

    def describe(self):
        return (f"{self.kind} on {self.topic} ({self.confirm} frames): "
                + "  ".join(f"{g}={b}" for g, b in GESTURE_BASE.items()) + "  | "
                + "  ".join(f"{g}={s.lstrip('_') or 'forward'}"
                            for g, s in GESTURE_SUFFIX.items()))

    def start(self):
        self._spawn(self._loop)

    def _loop(self):
        rclpy = self._rclpy
        owns_ros = not rclpy.ok()        # ServoPublisher inits unless --no_robot
        if owns_ros:
            rclpy.init()
        node = self._Node("gesture_gait")
        ex = self._Executor()
        ex.add_node(node)
        hands = self._mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=1,
            min_tracking_confidence=0.4, min_detection_confidence=0.4)

        def on_image(msg):
            # "rgb8" converts from whatever the camera publishes. MediaPipe
            # expects RGB; the demo passed it BGR.
            rgb = self._bridge.imgmsg_to_cv2(msg, "rgb8")
            res = hands.process(rgb)
            if not res.multi_hand_landmarks:
                self._observe(None)
                return
            h, w = rgb.shape[:2]
            pts = np.array([(p.x * w, p.y * h)
                            for p in res.multi_hand_landmarks[0].landmark])
            self._observe(classify_gesture(finger_angles(pts)))

        # Queue depth 1: frames that arrive while one is being processed are
        # dropped, so recognition never lags behind the camera.
        node.create_subscription(self._Image, self.topic, on_image, 1)
        try:
            while not self._stop.is_set():
                ex.spin_once(timeout_sec=0.1)
        finally:
            hands.close()
            ex.shutdown()
            node.destroy_node()
            if owns_ros:
                rclpy.shutdown()

    def _observe(self, gesture):
        """One frame's gesture (None = no hand)."""
        if gesture == self._last:
            self._run += 1
        else:
            self._last, self._run = gesture, 1
        if gesture is None:
            self._fired = None
        elif self._run == self.confirm and gesture != self._fired:
            self._fired = gesture
            self._apply(gesture)

    def _apply(self, gesture):
        base, suffix = split_gait(self.names[self.index])
        if gesture in GESTURE_BASE:
            b = GESTURE_BASE[gesture]
            candidates = [b + suffix, b]
        elif gesture in GESTURE_SUFFIX:
            candidates = [base + GESTURE_SUFFIX[gesture]]
        else:
            return                        # recognized, but mapped to nothing
        for name in candidates:
            if name in self.names:
                self.set_index(self.names.index(name), f"  [gesture {gesture}]")
                return
        print(f"  gesture {gesture}: no gait "
              + " or ".join(repr(n) for n in dict.fromkeys(candidates))
              + " in this config")


def make_gait_selectors(kinds, names, args):
    """-> (GaitState, [GaitSelector, ...]). 'none' adds no selector."""
    kinds = [k for k in dict.fromkeys(kinds) if k != "none"]
    if "joystick" in kinds:
        raise NotImplementedError(
            "--gait_switch joystick is not implemented. Subclass GaitSelector, "
            "override start() to spawn a listener that calls set_index(), and "
            "register it in make_gait_selectors -- the inference loop needs no "
            "changes.")
    schedule = None
    if "schedule" in kinds:
        if not args.gait_schedule:
            raise SystemExit("--gait_switch schedule needs --gait_schedule")
        schedule = parse_schedule(args.gait_schedule, names)
    # A schedule's cycle-0 gait is the starting gait, as it was before
    # selectors could be combined; otherwise --gait.
    state = GaitState(names, schedule[0][1] if schedule
                      else resolve_gait(args.gait, names))
    sels = []
    for kind in kinds:
        if kind == "schedule":
            sels.append(ScheduledGait(state, schedule, args.cycle_time))
        elif kind == "keyboard":
            sels.append(KeyboardGait(state, args.base_gait))
        elif kind == "gesture":
            sels.append(GestureGait(state, args.gesture_topic,
                                    args.gesture_frames))
    return state, sels


# ═══════════════════════════════════════════════════════════════════
# 3.  Servo output
# ═══════════════════════════════════════════════════════════════════

class ServoPublisher:
    """ROS2 ServosPosition publisher, or a no-op when --no_robot."""

    def __init__(self, servo_ids, enabled=True, duration=0.02, rate_hz=10.0):
        self.ids, self.enabled, self.duration = servo_ids, enabled, duration
        if not enabled:
            print("  --no_robot: commands computed but not published")
            return
        import rclpy
        from rclpy.node import Node
        from servo_controller_msgs.msg import ServoPosition, ServosPosition  # type: ignore
        self._ServoPosition, self._ServosPosition = ServoPosition, ServosPosition
        rclpy.init()
        self.node = Node("run_inference")
        self.pub = self.node.create_publisher(ServosPosition, "servo_controller", 1)
        print(f"  ROS2 publisher up on 'servo_controller' ({len(servo_ids)} servos)")

    def publish(self, values):
        if not self.enabled:
            return
        msg = self._ServosPosition()
        msg.duration = self.duration
        msg.position = [self._ServoPosition(id=i, position=float(v))
                        for i, v in zip(self.ids, values)]
        msg.position_unit = "pulse"
        self.pub.publish(msg)

    def close(self):
        if self.enabled:
            import rclpy
            self.node.destroy_node()
            rclpy.shutdown()


# ═══════════════════════════════════════════════════════════════════
# 4.  Inference loop
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def run(model, cfg, pub, state, sels, viz, args, device):
    n_joints = int(cfg_get(cfg, "n_joints", 8))
    lo = float(cfg_get(cfg, "global_min", -124.0))
    hi = float(cfg_get(cfg, "global_max", 124.0))
    scale, shift = (hi - lo) / 2.0, (hi + lo) / 2.0
    period = float(cfg_get(cfg, "cpg_period_steps", 254.0))
    names = list(cfg_get(cfg, "gait_names", []))

    # Timesteps have no inherent duration -- training only ever counted them.
    # cycle_time fixes the mapping: one gait cycle is `period` steps, so a step
    # is cycle_time/period seconds. This replaces the previous script's magic
    # sleep(0.07).
    dt = args.cycle_time / period
    print(f"\n  period {period:.0f} steps/cycle, cycle_time {args.cycle_time}s"
          f"  ->  dt {1e3*dt:.2f} ms/step ({1.0/dt:.0f} Hz)")
    print(f"  publishing every {args.publish_every} step(s) "
          f"({1.0/(dt*args.publish_every):.1f} Hz)")

    print(f"  starting gait: {names[state.index]}")
    for sel in sels:
        print(f"  gait switching: {sel.describe()}")
    if not sels:
        print("  gait switching: none (gait stays fixed)")

    cpg = CPGSource(cfg)
    state = model.init_state(1, device)
    gait_t = torch.zeros(1, dtype=torch.long, device=device)

    settle = int(args.settle_cycles * period)
    n_steps = settle + int(args.cycles * period) if args.cycles > 0 else -1
    print(f"  settling {settle} steps ({args.settle_cycles} cycles) before "
          f"publishing\n")

    lat, vlat = [], []
    t_start, step, started = time.perf_counter(), 0, False
    try:
        while n_steps < 0 or step < n_steps:
            # Selection lives entirely in the selectors' own threads; the
            # loop just reads whatever is current.
            gait_t.fill_(state.index)

            x = torch.as_tensor(cpg.step(), dtype=torch.float32,
                                device=device).unsqueeze(0)
            t0 = time.perf_counter()
            y, state, aux = model.step(x, gait_t, state)
            lat.append((time.perf_counter() - t0) * 1e3)
            joints = y[0].cpu().numpy() * scale + shift

            if step >= settle:
                if not started:
                    started = True
                    # Started here, not before settling, so schedule cycle 0
                    # means "first published cycle" rather than counting the
                    # settling steps.
                    for sel in sels:
                        sel.start()
                    print(f"  settled -- publishing\n")
                if step % args.publish_every == 0:
                    pub.publish(joints)
                if viz is not None:
                    # aux is (timing spikes,) for timing_grouped, None for
                    # dense. In process mode this is a queue put of a few
                    # hundred bytes; in inline mode it may redraw.
                    spk_t = (aux[0][0].cpu().numpy() if aux
                             else np.zeros(0, np.float32))
                    t_v = time.perf_counter()
                    viz.update(x[0].cpu().numpy(), spk_t, joints, state.index)
                    vlat.append((time.perf_counter() - t_v) * 1e3)

            step += 1
            # Sleep the remainder of this step's budget rather than a fixed
            # amount, so compute time does not accumulate as phase drift.
            slack = (t_start + step * dt) - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
    except KeyboardInterrupt:
        print("\n  [INTERRUPT] stopping")
    finally:
        for sel in sels:
            sel.stop()

    if vlat:
        v = np.array(vlat)
        print(f"\n  viz handoff cost per step (ms): mean {v.mean():.3f}  "
              f"p95 {np.percentile(v, 95):.3f}  max {v.max():.3f}")
        print(f"  (process mode should be well under 1 ms; anything near the "
              f"step budget means it is drawing inline)")
    if lat:
        l = np.array(lat)
        print(f"\n  model.step latency over {len(l)} calls (ms): "
              f"mean {l.mean():.3f}  median {np.median(l):.3f}  "
              f"p95 {np.percentile(l, 95):.3f}  max {l.max():.3f}")
        print(f"  budget per step {1e3*dt:.2f} ms -> "
              f"{'OK' if np.percentile(l, 95) < 1e3*dt else 'OVER BUDGET'}")
        real = time.perf_counter() - t_start
        print(f"  {step} steps in {real:.1f}s = {step/max(real,1e-9):.0f} Hz "
              f"(target {1.0/dt:.0f} Hz)")


def resolve_gait(g, names):
    """Accept a gait index or a name from the config's gait_names."""
    if g is None:
        return 0
    if g.isdigit():
        return int(g)
    if g not in names:
        raise SystemExit(f"Unknown gait {g!r}; config has {names}")
    return names.index(g)


def parse_schedule(s, names):
    """
    "0:tripod,8:ripple,16:1" -> [(0.0, 0), (8.0, 4), (16.0, 1)].

    Sorted by cycle, and an entry at cycle 0 is required so the starting gait
    is explicit rather than inherited from an unrelated flag.
    """
    out = []
    for part in s.split(","):
        if ":" not in part:
            raise SystemExit(f"--gait_schedule entry {part!r} is not cycle:gait")
        cyc, g = part.split(":", 1)
        out.append((float(cyc), resolve_gait(g.strip(), names)))
    out.sort()
    if not out or out[0][0] != 0.0:
        raise SystemExit("--gait_schedule must include an entry at cycle 0")
    return out


# ═══════════════════════════════════════════════════════════════════
# 5.  Entry point
# ═══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="CPG-SNN inference on the robot (PyTorch)")
    ap.add_argument("--model_dir", type=str, default="",
                    help="Resolved as outputs/<model_dir>, same as "
                         "visualize.py. Default '' = outputs/ itself.")
    ap.add_argument("--ckpt", type=str, default="best_model.pt")
    ap.add_argument("--cfg",  type=str, default="cpg_lif_snn_config.json")
    ap.add_argument("--gait_switch", type=str, nargs="+", default=["none"],
                    choices=["none", "schedule", "keyboard", "gesture",
                             "joystick"],
                    help="Where gait changes come from; several can be given "
                         "(e.g. 'keyboard gesture'), and the latest change "
                         "from any of them wins. 'none': hold --gait (index 0 "
                         "by default). 'schedule': follow --gait_schedule. "
                         "'keyboard': WASD on stdin, using --base_gait plus a "
                         "direction suffix. 'gesture': hand gestures from "
                         "--gesture_topic (see GestureGait). 'joystick' is a "
                         "placeholder and raises NotImplementedError.")
    ap.add_argument("--gesture_topic", type=str,
                    default="/depth_cam/rgb/image_raw",
                    help="For --gait_switch gesture: ROS Image topic to read.")
    ap.add_argument("--gesture_frames", type=int, default=5,
                    help="For --gait_switch gesture: consecutive frames a "
                         "gesture must be seen on before it acts.")
    ap.add_argument("--gait", type=str, default="0",
                    help="Starting gait, index or name from the config's "
                         "gait_names. Ignored when 'schedule' is among the "
                         "--gait_switch sources (its cycle-0 entry starts).")
    ap.add_argument("--gait_schedule", type=str, default=None,
                    help="For --gait_switch schedule: 'cycle:gait' pairs, e.g. "
                         "'0:tripod,8:ripple,16:0'. Cycles count from the "
                         "first published cycle, not from start-up.")
    ap.add_argument("--base_gait", type=str, default="tripod",
                    help="For --gait_switch keyboard: W uses this name, A/S/D "
                         "append _left/_backwards/_right. Any missing name is "
                         "reported at start-up and its key does nothing.")
    ap.add_argument("--cycle_time", type=float, default=2.0,
                    help="Wall-clock seconds per gait cycle. Sets the real "
                         "duration of a CPG timestep (cycle_time/period); "
                         "training only counted steps, so this is where the "
                         "physical timescale is chosen. Lower = faster gait.")
    ap.add_argument("--cycles", type=float, default=0,
                    help="Gait cycles to run after settling. 0 = forever "
                         "(Ctrl+C to stop).")
    ap.add_argument("--settle_cycles", type=float, default=3.0,
                    help="Cycles to step the model before publishing, so the "
                         "membranes are not at their zero-state transient "
                         "when the servos start following.")
    ap.add_argument("--publish_every", type=int, default=1,
                    help="Publish every Nth timestep. Raise if the servo "
                         "controller cannot keep up with 1/dt Hz.")
    ap.add_argument("--servo_ids", type=str, default=None,
                    help="Comma-separated servo ids in gait-table COLUMN "
                         "order. Default is the 18-servo hexapod map. Must be "
                         "n_joints long.")
    ap.add_argument("--duration", type=float, default=0.02,
                    help="ServosPosition.duration field (seconds).")
    ap.add_argument("--viz", action="store_true",
                    help="Open the live visualization window (see "
                         "live_visualization.py). Needs a display -- over SSH "
                         "that means 'ssh -X'. Runs in a separate PROCESS by "
                         "default, so its ~16 ms redraw cannot stall the "
                         "control loop; see --viz_mode.")
    ap.add_argument("--viz_mode", type=str, default="process",
                    choices=["process", "remote", "inline"],
                    help="'process' (default): the visualizer runs in a child "
                         "process fed by a bounded queue, so the control loop "
                         "only pays a few microseconds per frame and frames "
                         "are dropped rather than the robot being made to "
                         "wait. 'remote': same, but the figure is rendered "
                         "on another machine over a socket (see --viz_host) — "
                         "~7 kB/s of numbers instead of a framebuffer, which "
                         "is the way to watch it over SSH without the lag of "
                         "X11 forwarding or VNC. 'inline': draw inside the "
                         "control loop, which overruns a 16 ms step on an "
                         "18-joint hexapod; use only to debug the visualizer.")
    ap.add_argument("--viz_host", type=str, default="localhost:5555",
                    help="host:port for --viz_mode remote. With an SSH reverse "
                         "tunnel ('ssh -R 5555:localhost:5555 user@robot') the "
                         "default reaches the laptop. Start the viewer there "
                         "with 'python live_visualization.py --listen 5555'.")
    ap.add_argument("--viz_fps", type=float, default=12.0,
                    help="Redraw rate for --viz. Buffering still happens every "
                         "timestep; only drawing is throttled.")
    ap.add_argument("--viz_trace_cycles", type=float, default=5.0,
                    help="History shown in the joint-angle plots, in CPG "
                         "cycles. Expressed in cycles rather than timesteps so "
                         "it stays meaningful whether the period is 352 (real "
                         "oscillator) or 120 (--fake_cpg).")
    ap.add_argument("--viz_cpg_cycles", type=float, default=1.5,
                    help="History shown in the CPG spike plot, in CPG cycles. "
                         "Much shorter than --viz_trace_cycles on purpose: at "
                         "5 cycles the individual spikes in a burst merge into "
                         "a solid block, and seeing them is the point of that "
                         "plot.")
    ap.add_argument("--gaits_dir", type=str, default="../gaits",
                    help="Folder of {name}.csv gait tables, resolved as "
                         "this_file_dir/<gaits_dir>. Only needed for --viz, "
                         "which reads per-gait per-joint min/max from them to "
                         "scale the trace plot.")
    ap.add_argument("--no_robot", action="store_true",
                    help="Compute commands but publish nothing. No ROS "
                         "imports, so this runs anywhere.")
    ap.add_argument("--device", type=str, default=None,
                    help="cuda / cpu. Default: cuda if available.")
    args = ap.parse_args()

    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    this_dir = Path(__file__).resolve().parent
    model_dir = outputs_path(str(this_dir), args.model_dir)
    print(f"Device : {device}\nModel  : {model_dir}\n")

    print("[1/3] Loading checkpoint + config ...")
    cfg, model, arch = load_run(model_dir, args.ckpt, args.cfg, device)
    model.eval()

    n_joints = int(cfg_get(cfg, "n_joints", 8))
    ids = ([int(v) for v in args.servo_ids.split(",")] if args.servo_ids
           else HEXAPOD_SERVO_IDS[:n_joints] if n_joints <= len(HEXAPOD_SERVO_IDS)
           else None)
    if ids is None or len(ids) != n_joints:
        raise SystemExit(
            f"Need {n_joints} servo ids for this model's n_joints; got "
            f"{len(ids) if ids else 0}. Pass --servo_ids explicitly.")
    print(f"  arch={arch}  n_joints={n_joints}  "
          f"gate_mode={cfg_get(cfg, 'gate_mode', 'n/a')}  "
          f"fake_cpg={bool(cfg_get(cfg, 'fake_cpg', False))}")
    print(f"  column -> servo: {list(zip(range(n_joints), ids))}")

    print("\n[2/3] Output + gait switching ...")
    pub = ServoPublisher(ids, enabled=not args.no_robot,
                         duration=args.duration)
    state, sels = make_gait_selectors(args.gait_switch,
                             list(cfg_get(cfg, "gait_names", [])), args)

    viz = None
    if args.viz:
        # Imported lazily so matplotlib is not pulled in on the robot unless
        # the visualization is actually asked for. In "process" mode the parent
        # never imports a GUI backend at all -- only the child does.
        opts = dict(trace_cycles=args.viz_trace_cycles,
                    cpg_cycles=args.viz_cpg_cycles, fps=args.viz_fps)
        if args.viz_mode == "process":
            from live_visualization import VisualizerProxy
            viz = VisualizerProxy(cfg, this_dir / args.gaits_dir, **opts)
            print(f"  live visualization in a child process (pid "
                  f"{viz.proc.pid}); the control loop cannot be stalled by it")
        elif args.viz_mode == "remote":
            from live_visualization import VisualizerClient
            host, _, port = args.viz_host.partition(":")
            try:
                viz = VisualizerClient(cfg, host or "localhost",
                                       int(port or 5555))
            except OSError as e:
                raise SystemExit(
                    f"Could not reach the remote visualizer at "
                    f"{args.viz_host} ({e}). Start it on the viewing machine "
                    f"with 'python live_visualization.py --listen "
                    f"{port or 5555}', and if you are on SSH make sure the "
                    f"reverse tunnel is up: "
                    f"ssh -R {port or 5555}:localhost:{port or 5555} user@robot")
            print(f"  live visualization streaming to {args.viz_host} "
                  f"(~7 kB/s; rendering happens there, not here)")
        else:
            from live_visualization import LiveVisualizer
            viz = LiveVisualizer(cfg, this_dir / args.gaits_dir, **opts)
            print("  live visualization INLINE — redraws run inside the "
                  "control loop and will overrun the step budget")
        state.on_change = viz.notify_gait_switch

    print("\n[3/3] Running ...")
    try:
        run(model, cfg, pub, state, sels, viz, args, device)
    finally:
        pub.close()
        if viz is not None:
            viz.close()
    print("\nDone.")


if __name__ == "__main__":
    main()