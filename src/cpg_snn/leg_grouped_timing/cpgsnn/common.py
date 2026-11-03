"""
Shared utilities: the project folder, output routing (outputs/<run>/<category>/...), run-config lookups (cfg_get), JSON helpers, git info and joint names.
"""

import math
import os
import re
import subprocess
from pathlib import Path

import numpy as np


# The project folder, one level above this package. The scripts live there, and outputs/ and the
# gait folder resolve from it exactly as they did when everything was one folder.
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ═══════════════════════════════════════════════════════════════════
# Small utilities
# ═══════════════════════════════════════════════════════════════════

def json_safe(obj):
    """Recursively coerce numpy scalars/arrays, Paths and tuples to JSON types (a bare
    json.dump choking on a np.float32 would abort a run or lose its results).
    """
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        f = float(obj)
        return f if math.isfinite(f) else None      # NaN/inf are not valid JSON
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def git_info():
    """Short commit hash + dirty flag of the repo this file lives in."""
    here = PROJECT_DIR
    def run(cmd):
        return subprocess.check_output(
            cmd, cwd=here, stderr=subprocess.DEVNULL, timeout=5
        ).decode().strip()
    try:
        commit = run(["git", "rev-parse", "--short", "HEAD"])
        dirty  = bool(run(["git", "status", "--porcelain"]))
        return {"commit": commit, "dirty": dirty}
    except Exception:                                    # noqa: BLE001
        return {"commit": None, "dirty": None}


# Output files are grouped into per-category subfolders of the run dir, routed by FILENAME so
# every call site passes a bare name and the layout lives here (train.py,
# visualize.py and reorganize_outputs.py all consult this table). Left LOOSE:
# metrics.csv, training_curves.png, transition.png, rmse_heatmap.png. model/ entries are
# exact names (plus the epoch<N>.pt pattern) so nothing unexpected is swept in,
# including .onnx.data (only written with --export_onnx 1).
OUT_ROUTES = (
    # (kind, pattern, subfolder);  kind is "prefix" | "exact" | "regex"
    ("prefix", "recon_",                      "recons"),
    ("prefix", "timing_alignment_",           "timing_alignments"),
    ("prefix", "membranes_",                  "membrane_waveforms"),
    ("prefix", "phase_fold_",                 "phase_folds"),
    ("exact",  "best_model.pt",               "model"),
    ("regex",  r"epoch\d+\.pt",               "model"),
    ("exact",  "cpg_lif_snn_config.json",     "model"),
    ("exact",  "cpg_lif_snn_step.onnx",       "model"),
    ("exact",  "cpg_lif_snn_step.onnx.data",  "model"),
    ("exact",  "alignment_summary.png",       "misc_info"),
    ("exact",  "burst_threshold.png",         "misc_info"),
    ("exact",  "cpg_raster.png",              "misc_info"),
    ("exact",  "routing_matrices.png",        "misc_info"),
    ("exact",  "tau_distributions.png",       "misc_info"),
    # visualize.py suffixes these with the checkpoint name for any non-best checkpoint,
    # so they never overwrite best_model's.
    ("prefix", "timing_summary",              "misc_info"),
    ("prefix", "alignment_scores",            "misc_info"),
)
OUT_CATEGORY_DIRS = tuple(dict.fromkeys(d for _, _, d in OUT_ROUTES))


def route_subdir(name):
    """Category subfolder for a filename, or "" to leave it loose."""
    for kind, pat, sub in OUT_ROUTES:
        if ((kind == "prefix" and name.startswith(pat))
                or (kind == "exact" and name == pat)
                or (kind == "regex" and re.fullmatch(pat, name))):
            return sub
    return ""


def out_path(out_dir, name):
    """Routed path for WRITING `name`, with its parent directory created."""
    p = Path(out_dir, route_subdir(name), name)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def in_path(in_dir, name):
    """Routed path for READING `name`, falling back to the loose location (older runs
    keep best_model.pt at the top level). If neither exists the routed path is
    returned, so errors point at where the file should now be.
    """
    routed = Path(in_dir, route_subdir(name), name)
    if routed.exists():
        return routed
    loose = Path(in_dir, name)
    return loose if loose.exists() else routed

# Anatomical name of the k-th joint in a leg, keyed by joints per leg.
JOINT_TYPE_NAMES = {2: ["shoulder", "knee"],
                    3: ["coxa", "femur", "tibia"]}


def joint_type_names(k):
    """Names for k joints-per-leg, falling back to generic labels."""
    return JOINT_TYPE_NAMES.get(k, [f"joint {i}" for i in range(k)])

def outputs_path(this_file_dir, rel=""):
    """this_file_dir/outputs[/rel], so a bare name like "test1" lands at outputs/test1.
    rel="" resolves to outputs itself.
    """
    return Path(this_file_dir, "outputs", rel) if rel else Path(this_file_dir, "outputs")


# ═══════════════════════════════════════════════════════════════════
# Config lookups (used by visualize.py, inference.py)
# ═══════════════════════════════════════════════════════════════════

def cfg_get(cfg, key, default=None):
    """Look a key up at the config top level, then `model_detail`, then the verbatim `args`
    dump (the config records facts more than once on purpose: deployment contract,
    reproduction record, what was typed). Older configs lack newer keys, hence the default.
    """
    for src in (cfg, cfg.get("model_detail", {}), cfg.get("args", {})):
        if isinstance(src, dict) and src.get(key) is not None:
            return src[key]
    return default
