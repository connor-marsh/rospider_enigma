"""
Parameter sweep over train.py: one training run per combination of the varied args, the fixed
args held constant, run as parallel processes packed onto one GPU.

Usage
    python sweep.py --name lam --fixed "--n_cpg_neurons 18 --fake_cpg 1 --drive_source cpg_lif" \\
        --vary drive_lambda=1e-3,3e-3,1e-2 --vary seed=1,2,3
    python sweep.py --spec my_sweep.json          # {"name", "fixed", "vary"}; see load_spec
    python sweep.py ... --dry_run                 # print the plan, launch nothing

Layout: outputs/zsweeps/<name>/
    sweep.json         the spec, per-signature GPU estimates, and per run: params, command,
                       status, return code, start/end time
    runs.csv           one row per run: varied values, status, minutes, and results scraped
                       from each finished run's config (epochs, best val loss, mean RMSE)
    NNN_<arg>=<v>__<arg>=<v>/   each run's own out_dir (metrics.csv, model/, figures) plus
                       train.log, its stdout and stderr

GPU packing. Runs are grouped by their RESOURCE signature: the varied args that can change
memory or compute (any arg not in SAFE_ARGS). The first run of a signature is a probe: it starts
with the GPU to itself, and after its first epoch completes, its memory and power (above the idle
baseline) are sampled for --probe_seconds. The rest of that signature then run as a rolling pool,
a new run starting whenever one finishes, while the summed estimates stay under --target (default
80%) of both GPU memory and power headroom and live free memory confirms it. nvidia-smi's
utilization ("Volatile GPU-Util") is the fraction of TIME a kernel is running, not how much of the
GPU is in use, so it is shown but not used. A new signature waits for the GPU to drain before its
probe. --per_run_mem_mb / --per_run_power_w skip probing.

Throughput ramp. Without MPS, processes on one GPU are time-sliced: once it is busy ~100% of the
time, extra runs mostly slow each other down. So the pool grows ONE run at a time: once the newest
run's first epoch is done (always slow, so skipped), total throughput over its next --ramp_epochs
epochs is timed from each run's own per-epoch seconds. If the new run added less than
--min_marginal (default 50%) of an average run's throughput, the pool is held at the previous size
for that signature; otherwise another run is added, until the memory/power estimate or
--max_parallel stops it. The chosen pool is saved, so a resumed sweep skips the ramp.
--no_ramp fills by the estimate alone.

Ctrl+C (or SIGTERM) stops launching and sends SIGINT to the running runs, which train.py treats
as a normal finish (they are marked stopped_early); a second one kills them. Re-running the same
command resumes: only runs that completed every requested epoch are skipped.
"""
import argparse
import ast
import csv
import datetime
import difflib
import itertools
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

this_file_dir = os.path.dirname(os.path.abspath(__file__))

# Varied args that cannot change a run's GPU memory or compute per step (loss weights, schedules,
# seeds, init ranges, logging). Anything else gets its own probe per distinct value, which is
# safe but costs one probe. Extend as needed.
SAFE_ARGS = {
    "drive_lambda", "drive_penalty", "spike_stats_lambda", "spike_objective", "min_count_floor",
    "min_count_floor_weight", "lambda_hold", "lambda_warmup", "lr", "seed", "epochs",
    "grad_clip", "slope", "timing_slope", "tau_min", "tau_max", "tau_timing_max",
    "tau_readout_max", "tau_syn_max", "film_mode", "gate_mode", "bias_mode", "hidden_reset",
    "timing_reset", "calibrate_gains", "calibrate_per_gait", "reinit_dead_after", "sat_scale",
    "readout_gait_bias", "sub_film", "save_model_every", "timing_log_every", "log_every",
    "visualize", "export_onnx", "align_vel_tau", "align_cor_tau", "recon_cycles",
}


# ── spec ──────────────────────────────────────────────────────────────
def train_flags(*paths):
    """{name: kind} for every train.py flag (kind: 'value' | 'switch' | 'multi'), read from the
    source so a typo fails here, not N runs later. Includes the generated --cpg_<name> flags."""
    src = "\n".join(Path(p).read_text() for p in paths); tree = ast.parse(src); flags = {}
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "add_argument"
                and n.args and isinstance(n.args[0], ast.Constant) and str(n.args[0].value).startswith("--")):
            kw = {k.arg: k.value for k in n.keywords}
            act = kw.get("action")
            kind = ("switch" if isinstance(act, ast.Constant) and act.value == "store_true"
                    else "multi" if "nargs" in kw else "value")
            flags[n.args[0].value[2:]] = kind
    if "CPG_PARAMS = {" in src:
        blk = src[src.index("CPG_PARAMS = {"):]
        blk = blk[:blk.index("\n}") + 2]
        for name in re.findall(r'^\s+"([a-z_]+)":\s+\(', blk, re.M):
            flags[f"cpg_{name}"] = "value"
    return flags


def check_name(name, flags, where):
    if name not in flags:
        close = difflib.get_close_matches(name, flags, n=3, cutoff=0.5)
        hint = f" Did you mean {', '.join('--' + c for c in close)}?" if close else ""
        raise SystemExit(f"{where}: train.py has no --{name}.{hint}")


def load_spec(args, flags):
    """(name, fixed_tokens, vary) from --spec JSON and/or the CLI (CLI wins).
    JSON: {"name": str, "fixed": "--a 1 --b 2" | ["--a", "1"] | {"a": 1}, "vary": {"arg": [values]}}."""
    spec = {}
    if args.spec:
        p = Path(this_file_dir, args.spec)            # relative to this file; absolute paths as given
        if not p.exists():
            raise SystemExit(f"--spec: {p} not found (relative paths resolve from {this_file_dir})")
        spec = json.loads(p.read_text())
    name = args.name or spec.get("name")
    if not name:
        raise SystemExit("Give the sweep a --name (or a \"name\" in --spec).")
    fixed = spec.get("fixed", [])
    if isinstance(fixed, dict):
        fixed = [t for k, v in fixed.items() for t in (f"--{k}", str(v))]
    elif isinstance(fixed, str):
        fixed = shlex.split(fixed)
    fixed = list(fixed) + (shlex.split(args.fixed) if args.fixed else [])
    vary = {k.lstrip("-"): list(v) for k, v in spec.get("vary", {}).items()}
    for item in args.vary or []:
        if "=" not in item:
            raise SystemExit(f"--vary {item!r}: expected name=v1,v2,...")
        k, vals = item.split("=", 1)
        vary[k.lstrip("-")] = [v.strip() for v in vals.split(",") if v.strip()]
    fixed_names = [t[2:] for t in fixed if t.startswith("--")]
    for k in fixed_names:
        check_name(k, flags, "--fixed")
    for k, vals in vary.items():
        check_name(k, flags, "--vary")
        if not vals:
            raise SystemExit(f"--vary {k}: no values")
        if k in fixed_names:
            raise SystemExit(f"--{k} is both fixed and varied")
    for k in ("out_dir",):
        if k in fixed_names or k in vary:
            raise SystemExit(f"--{k} is set by the sweep; remove it")
    return name, fixed, vary


def tokens(name, value, kind):
    if kind == "switch":
        return [f"--{name}"] if str(value).lower() in ("1", "true", "yes", "on") else []
    if kind == "multi":
        vals = value if isinstance(value, list) else shlex.split(str(value))
        return [f"--{name}", *map(str, vals)]
    return [f"--{name}", str(value)]


def plan_runs(vary):
    """All combinations, in the order given, grouped by resource signature (first-seen order)."""
    keys = list(vary)
    combos = list(itertools.product(*(vary[k] for k in keys))) if keys else [()]
    runs = []
    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        label = "__".join(f"{k}={re.sub(r'[^A-Za-z0-9.+-]', '-', str(v))}" for k, v in params.items())
        sig = json.dumps(sorted((k, str(v)) for k, v in params.items() if k not in SAFE_ARGS))
        runs.append({"index": i, "name": f"{i:03d}_{label or 'base'}", "params": params, "sig": sig})
    order = {s: j for j, s in enumerate(dict.fromkeys(r["sig"] for r in runs))}
    return sorted(runs, key=lambda r: (order[r["sig"]], r["index"]))


# ── GPU ───────────────────────────────────────────────────────────────
class GPU:
    """nvidia-smi for one physical GPU. .ok: memory readable; .power_ok: power readable (some
    virtualized / MIG GPUs report [N/A], which leaves memory as the only packing limit)."""
    def __init__(self, index):
        self.index = index
        q = self.query() if shutil.which("nvidia-smi") else None
        self.ok = q is not None
        self.power_ok = bool(q) and q["power"] is not None and q["plimit"] is not None

    def query(self):
        """{util, mem, total, power, plimit} (%, MB, MB, W, W) or None. `util` is nvidia-smi's
        "Volatile GPU-Util": the fraction of TIME a kernel was running, not how much of the GPU
        it used, so it is displayed but never used for packing."""
        try:
            out = subprocess.run(
                ["nvidia-smi", "-i", str(self.index),
                 "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,power.limit",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15).stdout
            f = [x.strip() for x in out.strip().splitlines()[0].split(",")]
            num = lambda x: float(x) if re.fullmatch(r"[\d.]+", x) else None
            q = dict(zip(("util", "mem", "total", "power", "plimit"), map(num, f)))
            return q if q["mem"] is not None and q["total"] is not None else None
        except Exception:
            return None


# ── bookkeeping ───────────────────────────────────────────────────────
def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def write_json(path, obj):
    tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(obj, indent=2)); os.replace(tmp, path)


def scrape(run_dir):
    """Result fields from a run's config; {} if it has none yet."""
    for p in (run_dir / "model" / "cpg_lif_snn_config.json", run_dir / "cpg_lif_snn_config.json"):
        if p.exists():
            try:
                c = json.loads(p.read_text())
            except ValueError:
                return {}
            tr = c.get("training", {}) or {}
            rm = c.get("per_joint_rmse_deg") or []
            return {"config_status": c.get("status"), "epochs_completed": tr.get("epochs_completed"),
                    "epochs_requested": tr.get("epochs_requested"),
                    "best_val_mse": tr.get("best_val_mse"),
                    "rmse_mean_deg": (sum(rm) / len(rm)) if rm else None}
    return {}


def finished(run_dir):
    """Config says complete AND every requested epoch ran: train.py also writes "complete"
    after a Ctrl+C, which must not count as done."""
    r = scrape(run_dir)
    if r.get("config_status") != "complete":
        return False
    req, got = r.get("epochs_requested"), r.get("epochs_completed")
    return req is None or got is None or got >= req


def write_summary(root, manifest):
    keys = list(manifest["vary"])
    cols = ["index", "name", *keys, "status", "returncode", "minutes", "epochs_completed",
            "best_val_mse", "rmse_mean_deg"]
    with open(root / "runs.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(cols)
        for nm, r in sorted(manifest["runs"].items(), key=lambda kv: kv[1]["index"]):
            res = scrape(root / nm)
            mins = (round((datetime.datetime.fromisoformat(r["end"]) -
                           datetime.datetime.fromisoformat(r["start"])).total_seconds() / 60, 2)
                    if r.get("start") and r.get("end") else "")
            w.writerow([r["index"], nm, *[r["params"][k] for k in keys], r["status"],
                        r.get("returncode", ""), mins, *[res.get(k, "") for k in
                        ("epochs_completed", "best_val_mse", "rmse_mean_deg")]])


# ── live progress ─────────────────────────────────────────────────────
# Markers train.py (and the visualization it runs) prints; the LAST one in a run's log
# is its current stage.
STAGES = [
    (r"\[0/6\]", "loading gaits"), (r"\[1/6\]", "running CPG"), (r"\[2/6\]", "analyzing CPG"),
    (r"\[[34]/6\]", "preparing data"), (r"\[5/6\]", "building model"),
    (r"Calibrating timing-layer gains", "calibrating"),
    (r"Per-epoch metrics -> metrics\.csv", "training"),
    (r"\[6/6\]", "evaluating"), (r"\[7/7\]", "visualizing"), (r"\[([1-5])/5\]", "visualizing"),
    (r"(?m)^Done \u2014", "finishing"),
]
AFTER_TRAINING = {"evaluating", "visualizing", "finishing"}


def update_stage(p, log_path):
    """Read the new part of a run's log and advance p["stage"] (and p["vis"], the k of k/5)."""
    try:
        with open(log_path, "rb") as fh:
            fh.seek(p["offset"]); new = fh.read().decode("utf-8", "replace"); p["offset"] = fh.tell()
    except OSError:
        return
    text = p["tail"] + new; p["tail"] = text[-300:]           # a marker may straddle two reads
    best = None
    for pat, stage in STAGES:
        for m in re.finditer(pat, text):
            if best is None or m.start() >= best[0]:
                best = (m.start(), stage, m.group(1) if m.groups() else None)
    if best:
        p["stage"] = best[1]
        if best[2]:
            p["vis"] = best[2]
        if best[1] == "training" and p["t_train"] is None:
            p["t_train"] = time.time()


def metrics_rows(p, path):
    """Epochs logged so far: rows of metrics.csv (re-counted only when the file grows)."""
    try:
        size = os.path.getsize(path)
    except OSError:
        return 0
    if size != p["msize"]:
        with open(path, "rb") as fh:
            p["mrows"] = max(fh.read().count(b"\n") - 1, 0)
        p["msize"] = size
    return p["mrows"]


def epoch_secs(p, path):
    """Append (epoch, sec) for rows added to a run's metrics.csv since the last call and return
    them all. `sec` is train.py's per-epoch train+val time, so throughput comes from the run's
    own timing rather than from polling."""
    eps = p.setdefault("eps", [])
    try:
        with open(path, "rb") as fh:
            fh.seek(p.get("m_off", 0)); data = fh.read()
    except OSError:
        return eps
    if not data:
        return eps
    p["m_off"] = p.get("m_off", 0) + len(data)
    lines = (p.get("m_buf", "") + data.decode("utf-8", "replace")).split("\n")
    p["m_buf"] = lines.pop()                                  # the last piece may be partial
    for ln in lines:
        cells = ln.strip().split(",")
        if "m_cols" not in p:
            p["m_cols"] = {c: i for i, c in enumerate(cells)}
            continue
        try:
            eps.append((int(float(cells[p["m_cols"]["epoch"]])), float(cells[p["m_cols"]["sec"]])))
        except (KeyError, ValueError, IndexError):
            pass
    return eps


def hms(sec):
    sec = int(max(sec, 0)); h, r = divmod(sec, 3600); m, s = divmod(r, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


class Display:
    """A block of status lines redrawn in place on a terminal (ANSI cursor-up + clear), with
    event messages printed above it. When stdout is not a terminal (a log file), the block is
    printed every `every` seconds instead, without escape codes."""
    def __init__(self, mode, every):
        self.live = mode == "on" or (mode == "auto" and sys.stdout.isatty())
        self.every, self.h, self.lines, self.t_last = every, 0, [], 0.0
        utf = "utf" in (sys.stdout.encoding or "").lower()
        self.full, self.empty = ("\u2588", "\u2591") if utf else ("#", "-")

    def bar(self, frac, n=20):
        k = int(round(max(0.0, min(1.0, frac)) * n))
        return self.full * k + self.empty * (n - k)

    def _clear(self):
        if self.h:
            sys.stdout.write(f"\x1b[{self.h}A\r\x1b[J"); self.h = 0

    def _draw(self):
        w = max(shutil.get_terminal_size((120, 24)).columns - 1, 20)
        for line in self.lines:                     # never wrap: wrapping breaks the redraw
            sys.stdout.write(line[:w] + "\n")
        self.h = len(self.lines)

    def log(self, msg):
        if self.live:
            self._clear(); sys.stdout.write(f"{msg}\n"); self._draw()
        else:
            sys.stdout.write(f"{msg}\n")
        sys.stdout.flush()

    def update(self, lines):
        self.lines = lines
        if self.live:
            self._clear(); self._draw()
        elif lines and time.time() - self.t_last >= self.every:
            self.t_last = time.time(); sys.stdout.write("\n".join([f"[{now()}]"] + lines) + "\n")
        sys.stdout.flush()

    def close(self):
        if self.live:
            self._clear()
        self.lines = []; sys.stdout.flush()


# ── main ──────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--name", help="Sweep folder name under outputs/zsweeps/.")
    ap.add_argument("--spec", help="JSON spec with name / fixed / vary, relative to this file "
                                   "(CLI values are added).")
    ap.add_argument("--fixed", help='Args every run gets, as on a command line: "--hidden 64 --fake_cpg 1".')
    ap.add_argument("--vary", action="append", help="name=v1,v2,... (repeatable); every combination runs.")
    ap.add_argument("--train_script", default="train.py", help="Relative to this file.")
    ap.add_argument("--gpu", type=int, default=0, help="Physical GPU index (sets CUDA_VISIBLE_DEVICES).")
    ap.add_argument("--target", type=float, default=0.80, help="Fraction of GPU memory and utilization to fill.")
    ap.add_argument("--max_parallel", type=int, default=None,
                    help="Cap on concurrent runs (default 16 with a GPU monitor, 1 without).")
    ap.add_argument("--probe_seconds", type=float, default=90.0,
                    help="How long to sample a probe after its first epoch completes.")
    ap.add_argument("--probe_timeout", type=float, default=1800.0,
                    help="Give up waiting for a probe to become active after this long.")
    ap.add_argument("--mem_margin", type=float, default=1.25, help="Safety factor on the memory estimate.")
    ap.add_argument("--power_margin", type=float, default=1.20, help="Safety factor on the power estimate.")
    ap.add_argument("--per_run_mem_mb", type=float, default=None, help="Skip probing: memory per run (MB).")
    ap.add_argument("--per_run_power_w", type=float, default=None,
                    help="Skip probing: power per run above idle (W).")
    ap.add_argument("--ramp_epochs", type=int, default=5,
                    help="Throughput ramp: epochs of the newest run to time (after skipping its "
                         "first) before deciding whether to add another.")
    ap.add_argument("--min_marginal", type=float, default=0.5,
                    help="Stop growing the pool when a new run adds less than this fraction of an "
                         "average run's throughput (1.0 = perfect scaling, 0 = no gain).")
    ap.add_argument("--no_ramp", action="store_true",
                    help="Skip the throughput ramp; fill the GPU by the memory/power estimate alone.")
    ap.add_argument("--poll", type=float, default=5.0, help="Seconds between scheduler checks.")
    ap.add_argument("--stagger", type=float, default=15.0,
                    help="Seconds between launches, so each run's memory shows up before the next.")
    ap.add_argument("--progress", choices=("auto", "on", "off"), default="auto",
                    help="Live per-run status block: auto = only when stdout is a terminal.")
    ap.add_argument("--status_every", type=float, default=60.0,
                    help="Without a terminal, print the status block every N seconds.")
    ap.add_argument("--dry_run", action="store_true", help="Print the plan and launch nothing.")
    args = ap.parse_args()
    disp = Display(args.progress, args.status_every)

    train_py = Path(this_file_dir, args.train_script)
    if not train_py.exists():
        raise SystemExit(f"{train_py} not found")
    flags = train_flags(Path(this_file_dir, "submodules", "train_args.py"),
                        Path(this_file_dir, "submodules", "cpg.py"))
    name, fixed, vary = load_spec(args, flags)
    runs = plan_runs(vary)
    root = Path(this_file_dir, "outputs", "zsweeps", name)
    rel = lambda r: f"zsweeps/{name}/{r['name']}"
    cmd_of = lambda r: [sys.executable, str(train_py), *fixed,
                        *[t for k, v in r["params"].items() for t in tokens(k, v, flags[k])],
                        "--out_dir", rel(r)]
    sigs = list(dict.fromkeys(r["sig"] for r in runs))

    disp.log(f"Sweep '{name}': {len(runs)} run(s), {len(sigs)} resource signature(s) -> {root}")
    for k, v in vary.items():
        disp.log(f"  vary --{k}: {v}" + ("" if k in SAFE_ARGS else "   (resource-relevant: probed per value)"))
    if args.dry_run:
        for r in runs:
            disp.log(f"  {r['name']}\n      {shlex.join(cmd_of(r)[1:])}")
        return

    spec = {"fixed": fixed, "vary": vary, "train_script": args.train_script}
    root.mkdir(parents=True, exist_ok=True)
    mpath = root / "sweep.json"
    if mpath.exists():
        manifest = json.loads(mpath.read_text())
        if {k: manifest.get(k) for k in spec} != spec:
            raise SystemExit(f"{root} already holds a sweep with a different spec; use a new --name.")
        disp.log(f"  resuming ({sum(finished(root / r['name']) for r in runs)} already complete)")
    else:
        manifest = {**spec, "name": name, "created": now(), "estimates": {},
                    "runs": {r["name"]: {"index": r["index"], "params": r["params"], "sig": r["sig"],
                                         "cmd": shlex.join(cmd_of(r)[1:]), "status": "pending"}
                             for r in runs}}
    est = manifest.setdefault("estimates", {})
    pending = [r for r in runs if not finished(root / r["name"])]
    for r in runs:
        if r not in pending:
            manifest["runs"][r["name"]]["status"] = "complete"
    write_json(mpath, manifest)

    gpu = GPU(args.gpu)
    max_par = args.max_parallel or (16 if gpu.ok else 1)
    if not gpu.ok:
        disp.log(f"  no nvidia-smi: running up to {max_par} at a time without GPU monitoring")
    override = args.per_run_mem_mb is not None or args.per_run_power_w is not None
    usable = lambda e: bool(e) and "mem_mb" in e and (not gpu.power_ok or "power_w" in e)
    idle_q = [q for q in (gpu.query() for _ in range(3)) if q] if gpu.ok else []
    idle_power = min(q["power"] for q in idle_q) if gpu.power_ok and idle_q else None
    if gpu.ok and not gpu.power_ok:
        disp.log("  GPU power is not readable here: packing by memory only, so set --max_parallel "
                 "to bound compute")
    running, stopping = {}, {"n": 0}

    def on_signal(signum, frame):
        stopping["n"] += 1
        if stopping["n"] == 1:
            disp.log(f"\n[{now()}] stopping: no new launches; SIGINT sent to {len(running)} run(s) "
                  f"(signal again to kill them)")
            for p in running.values():
                try: os.killpg(p["proc"].pid, signal.SIGINT)
                except ProcessLookupError: pass
        else:
            for p in running.values():
                try: os.killpg(p["proc"].pid, signal.SIGKILL)
                except ProcessLookupError: pass
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    def launch(r):
        d = root / r["name"]; d.mkdir(parents=True, exist_ok=True)
        log = open(d / "train.log", "a"); log.write(f"\n===== {now()} {shlex.join(cmd_of(r)[1:])}\n"); log.flush()
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        if gpu.ok:
            env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        proc = subprocess.Popen(cmd_of(r), cwd=this_file_dir, stdout=log, stderr=subprocess.STDOUT,
                                env=env, start_new_session=True)
        running[r["name"]] = {"proc": proc, "log": log, "run": r, "est": est.get(r["sig"]),
                              "t0": time.time(), "offset": 0, "tail": "", "stage": "starting",
                              "vis": None, "t_train": None, "ep_total": None, "msize": -1,
                              "mrows": 0, "probe": False}
        m = manifest["runs"][r["name"]]; m.update(status="running", start=now()); m.pop("end", None)
        write_json(mpath, manifest)
        disp.log(f"[{now()}] launched {r['name']}  ({len(running)} running)")

    def reap():
        for nm in [n for n, p in running.items() if p["proc"].poll() is not None]:
            p = running.pop(nm); p["log"].close(); rc = p["proc"].returncode
            cs = scrape(root / nm).get("config_status")
            status = ("complete" if rc == 0 and finished(root / nm) else
                      "stopped_early" if rc == 0 and cs == "complete" else
                      "interrupted" if stopping["n"] else "failed" if rc != 0 else "incomplete")
            manifest["runs"][nm].update(status=status, returncode=rc, end=now())
            write_json(mpath, manifest); write_summary(root, manifest)
            disp.log(f"[{now()}] {status:<11} {nm} (rc {rc}, {len(running)} running)")

    def probe(r):
        """Launch r with the GPU to itself and wait for its FIRST EPOCH to complete (a row in its
        metrics.csv): by then setup, calibration and the first-epoch compile are past, and training
        is at its steady load. Then sample for --probe_seconds. Estimate = peak memory and mean
        power, each above the pre-launch idle baseline and times its margin."""
        base = [q for q in (gpu.query() for _ in range(3)) if q]
        idle_mem = max(q["mem"] for q in base)
        idle_pw = min(q["power"] for q in base) if gpu.power_ok else None
        launch(r); nm = r["name"]; p = running[nm]; p["probe"] = True; proc = p["proc"]
        t0, started, samples = time.time(), None, []
        what = r["sig"] if r["sig"] != "[]" else "(no size-changing args)"
        disp.log(f"  probing {what} with {nm}: waiting for its first epoch, then sampling "
                 f"{args.probe_seconds:.0f}s")
        while time.time() - t0 < args.probe_timeout and proc.poll() is None and not stopping["n"]:
            if started is None:
                update_stage(p, root / nm / "train.log")
                long_epoch = p["t_train"] is not None and time.time() - p["t_train"] > 600
                if metrics_rows(p, root / nm / "metrics.csv") >= 1 or long_epoch:
                    started = time.time()                # (or 10 min into a very long first epoch)
                    disp.log(f"  {nm}: " + ("10 min into a long first epoch" if long_epoch and
                             not p["mrows"] else "first epoch done") +
                             f" after {hms(started - t0)}; sampling memory and power")
            else:
                q = gpu.query()
                if q:
                    samples.append((q["mem"] - idle_mem,
                                    q["power"] - idle_pw if idle_pw is not None and q["power"] is not None else None,
                                    q["util"]))
                if time.time() - started >= args.probe_seconds and len(samples) >= 3:
                    break
            idle(args.poll)
        if samples:
            mem = max(max(x[0] for x in samples), 1.0)
            pws = [x[1] for x in samples if x[1] is not None]
            us = sorted(x[2] for x in samples if x[2] is not None)
            e = {"mem_mb": mem * args.mem_margin,
                 "power_w": max(sum(pws) / len(pws), 1.0) * args.power_margin if pws else None,
                 "util": us[int(0.9 * (len(us) - 1))] if us else None}
        else:
            why = (f"exited (rc {proc.returncode}) before finishing an epoch; see its train.log"
                   if proc.poll() is not None else "finished no epoch within --probe_timeout")
            disp.log(f"  probe {why}: running this signature one at a time")
            e = {"mem_mb": float("inf"), "power_w": float("inf"), "util": None}
        est[r["sig"]] = e; p.update(est=e, probe=False); write_json(mpath, manifest)
        q = gpu.query()
        k_mem = args.target * q["total"] / e["mem_mb"] if q else max_par
        k_pw = (args.target * (q["plimit"] - idle_power) / e["power_w"]
                if q and gpu.power_ok and e["power_w"] and idle_power is not None else float("inf"))
        k = int(min(k_mem, k_pw, max_par))
        limit = "memory" if k_mem <= min(k_pw, max_par) else "power" if k_pw <= max_par else "--max_parallel"
        pw_txt = f", {e['power_w']:.0f} W" if e.get("power_w") not in (None, float("inf")) else ""
        ut_txt = f" (busy {e['util']:.0f}% of the time alone)" if e.get("util") is not None else ""
        disp.log(f"  estimate: {e['mem_mb']:.0f} MB{pw_txt} per run{ut_txt} -> about {k} at a time, "
                 f"limited by {limit}")

    gcache = {"t": 0.0, "q": None}

    def show():
        if not (disp.live or time.time() - disp.t_last >= disp.every):
            return                                   # nothing to draw yet; skip the file reads
        if gpu.ok and time.time() - gcache["t"] > 2.0:
            gcache.update(t=time.time(), q=gpu.query())
        nw = min(max((len(n) for n in running), default=10), 44)
        lines, frac_sum, rate_sum = [], 0.0, 0.0
        for nm, p in sorted(running.items(), key=lambda kv: kv[1]["run"]["index"]):
            d = root / nm
            update_stage(p, d / "train.log")
            st, pct, detail = p["stage"], 1.0, ""
            if st == "training":
                if p["ep_total"] is None:
                    p["ep_total"] = scrape(d).get("epochs_requested")
                k, n_ep = metrics_rows(p, d / "metrics.csv"), p["ep_total"]
                hist = p.setdefault("hist", []); t_now = time.time()
                hist.append((t_now, k)); hist[:] = [h for h in hist if t_now - h[0] <= 300]
                if len(hist) >= 2 and hist[-1][1] > hist[0][1]:   # epochs/min over the last 5 min
                    rate_sum += (hist[-1][1] - hist[0][1]) / (hist[-1][0] - hist[0][0]) * 60
                if n_ep:
                    k = min(k, n_ep); pct = 1 + 98 * k / n_ep; detail = f"ep {k}/{n_ep}"
                    if k and p["t_train"]:
                        detail += f"  eta {hms((time.time() - p['t_train']) / k * (n_ep - k))}"
                else:
                    detail = f"ep {k}"
            elif st in AFTER_TRAINING:
                pct = 99.0
                if st == "visualizing" and p["vis"]:
                    st = f"visualizing {p['vis']}/5"
            frac_sum += pct / 100
            tag = "  (probe)" if p["probe"] else ""
            lines.append(f"  {nm[:nw]:<{nw}}  {st:<16} {disp.bar(pct / 100)} {pct:5.1f}%  "
                         f"{hms(time.time() - p['t0']):>7}  {detail}{tag}")
        done = sum(r["status"] not in ("pending", "running", "not started")
                   for r in manifest["runs"].values())
        overall = (done + frac_sum) / max(len(runs), 1)
        head = (f"sweep {name}: {done}/{len(runs)} finished, {len(running)} running, "
                f"{len(pending)} queued  {disp.bar(overall, 24)} {overall * 100:5.1f}%")
        q = gcache["q"]
        if q:
            head += f"   GPU busy {q['util']:.0f}%" if q["util"] is not None else "   GPU"
            if q["power"] is not None and q["plimit"] is not None:
                head += f"  {q['power']:.0f}/{q['plimit']:.0f} W"
            head += f"  {q['mem'] / 1024:.1f}/{q['total'] / 1024:.0f} GB"
        if rate_sum:
            head += f"   {rate_sum:.2f} ep/min total"
        disp.update([head] + lines)

    def idle(seconds):
        """Sleep, refreshing the status block about once a second."""
        end = time.time() + seconds
        while True:
            show()
            left = end - time.time()
            if left <= 0:
                return
            time.sleep(min(1.0, left))

    ramp = {}                                            # sig -> throughput-ramp state

    def group(sig):
        return [nm for nm, p in running.items() if p["run"]["sig"] == sig]

    def ramp_begin(sig, newest):
        st = ramp.setdefault(sig, {"T": [], "restarts": 0, "pool": None})
        st.update(newest=newest, start=None, members=None)

    def conclude(sig, pool, why):
        ramp[sig]["pool"] = pool
        est.setdefault(sig, {})["pool"] = pool; write_json(mpath, manifest)
        disp.log(f"  ramp: holding the pool at {pool} run(s) for this signature ({why})")

    def ramp_step(sig):
        """Advance the throughput measurement for `sig`: 'wait', 'grow' or 'held'."""
        st = ramp[sig]; members = group(sig)
        for nm in members:
            epoch_secs(running[nm], root / nm / "metrics.csv")
            update_stage(running[nm], root / nm / "train.log")
        new = running.get(st["newest"])
        if new is None:                                  # the newest run ended before we could time it
            conclude(sig, max(len(members), 1), "the newest run ended before it could be timed")
            return "held"
        if st["start"] is None:
            if len(new["eps"]) < 1:                      # its first epoch is always slow: skip it
                return "wait"
            st.update(start={nm: len(running[nm]["eps"]) for nm in members}, members=set(members))
            return "wait"
        left = [nm for nm in st["members"] if nm not in running or running[nm]["stage"] in AFTER_TRAINING]
        if set(members) != st["members"] or left:        # the pool changed mid-window: re-time it
            st["restarts"] += 1; st["start"] = None
            if st["restarts"] > 3:
                conclude(sig, len(members), "runs keep finishing before a measurement completes")
                return "held"
            return "wait"
        if len(new["eps"]) - st["start"][st["newest"]] < args.ramp_epochs:
            return "wait"
        total = 0.0
        for nm in st["members"]:
            ep = running[nm]["eps"][st["start"][nm]:]
            if ep:
                total += 60.0 / (sum(x[1] for x in ep) / len(ep))
        k = len(st["members"]); st["T"].append((k, total)); st["restarts"] = 0
        if k == 1:
            disp.log(f"  ramp: 1 run -> {total:.2f} ep/min")
            return "grow"
        pk, pt = st["T"][-2]
        marginal = (total - pt) / (pt / pk) if pt > 0 else 1.0
        share = f"{marginal:.0%}" if abs(marginal) >= 0.005 else "0%"
        disp.log(f"  ramp: {k} runs -> {total:.2f} ep/min total ({total - pt:+.2f}; the new run added "
                 f"{share} of an average run's throughput)")
        if marginal < args.min_marginal:
            conclude(sig, pk, f"run {k} added under {args.min_marginal:.0%} of a run: the GPU is "
                              f"saturated, so more runs would only slow each other down")
            return "held"
        return "grow"

    def fits(e):
        if not running:
            return True                                  # never deadlock on an oversized run
        if len(running) >= max_par:
            return False
        if not gpu.ok:
            return True
        q = gpu.query()
        if q is None:
            return False
        ests = [p["est"] for p in running.values() if p["est"]]
        budget_mem = args.target * q["total"]
        ok = (sum(x["mem_mb"] for x in ests) + e["mem_mb"] <= budget_mem
              and q["mem"] + e["mem_mb"] <= budget_mem)            # estimates AND live free memory
        if ok and gpu.power_ok and idle_power is not None and e.get("power_w") is not None:
            budget_pw = args.target * (q["plimit"] - idle_power)
            ok = sum(x.get("power_w") or 0.0 for x in ests) + e["power_w"] <= budget_pw
        return ok

    disp.log(f"  {len(pending)} to run, up to {max_par} at a time, target {args.target:.0%} of the GPU")
    while pending or running:
        reap()
        if pending and not stopping["n"]:
            r = pending[0]
            if override and r["sig"] not in est:
                est[r["sig"]] = {"mem_mb": args.per_run_mem_mb or 0.0,
                                 "power_w": args.per_run_power_w or 0.0, "util": None}
            sig = r["sig"]
            want_ramp = not args.no_ramp and "pool" not in est.get(sig, {})
            st = ramp.get(sig)
            if gpu.ok and not usable(est.get(sig)):         # unknown (or pre-power) estimate
                if not running:                          # a new signature probes alone
                    probe(pending.pop(0))
                    if want_ramp and r["name"] in running:
                        ramp_begin(sig, r["name"])
            elif st and st["pool"] is None:              # ramping: time, then grow or hold
                if ramp_step(sig) == "grow":
                    if fits(est.get(sig) or {"mem_mb": 0.0}):
                        launch(pending.pop(0)); ramp_begin(sig, r["name"])
                        idle(args.stagger)
                        continue
                    conclude(sig, len(group(sig)), "the memory/power estimate or --max_parallel "
                                                   "allows no more")
            else:
                cap = (st or {}).get("pool") or est.get(sig, {}).get("pool") or float("inf")
                if len(group(sig)) < cap and fits(est.get(sig) or {"mem_mb": 0.0}):
                    launch(pending.pop(0))
                    if want_ramp and sig not in ramp:
                        ramp_begin(sig, r["name"])
                    idle(args.stagger)
                    continue
        elif stopping["n"] and not running:
            break
        idle(args.poll)
    disp.close()

    for r in pending:
        manifest["runs"][r["name"]]["status"] = "not started"
    write_json(mpath, manifest); write_summary(root, manifest)
    done = [r for r in manifest["runs"].values() if r["status"] == "complete"]
    disp.log(f"\nDone: {len(done)}/{len(runs)} complete. Summary: {root / 'runs.csv'}")


if __name__ == "__main__":
    main()