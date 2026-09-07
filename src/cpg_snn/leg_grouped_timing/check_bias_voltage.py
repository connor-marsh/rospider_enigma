"""
Can a sub-network hidden unit spike on a timestep where its timing neuron
did not?  Answered from the checkpoint alone -- no CPG replay, no forward
pass, no GPU.
=====================================================================

Only meaningful for `--arch timing_grouped --bias_mode voltage`.

The argument
------------
Under `bias_mode="voltage"` the injection into sub-net layer 1 is

    cur1 = (spk_t * w1) * film1_gamma + film1_beta        # film1 optional
    th1  = thresh - v1

On a timestep where the timing neuron did NOT spike, `spk_t == 0`, so the
first term vanishes and the ONLY drive left is `film1_beta`.  (If `--sub_ln`
covers this layer, LayerNorm sits between the two terms, but LN of an
all-zero vector is all-zero, so it changes nothing here.)  Layer 2 is the
same story with `spk1` in place of `spk_t` and `film2_beta` in place of
`film1_beta`.

That gives four possible states per unit, per gait:

    IMPOSSIBLE   beta <= 0 and th > 0.  A positive membrane can only decay
                 toward 0 from above, and `spike_fn` uses a strict `>`, so
                 the unit cannot fire without a timing spike.  Full stop.
    ASSIST       beta > 0, but the steady state it can reach on its own,
                 beta/(1-decay), is <= th.  Cannot fire unaided, but it
                 adds drive, so leftover membrane from the last timing
                 spike can cross threshold a few steps later.
    SPONTANEOUS  beta/(1-decay) > th.  Fires on a fixed period with no
                 timing input whatsoever.  The timing layer is decorative
                 for this unit.
    ALWAYS       th <= 0, i.e. v >= thresh.  Fires on EVERY timestep and
                 cannot recover: reset-to-zero lands at 0, which is still
                 above a non-positive threshold, and subtractive reset is
                 the same failure one step slower.

`train.py` now clamps v into [-thresh, thresh - 1e-6] after each optimiser
step, so ALWAYS should be extinct in new runs.  It is still reported here
because pre-clamp checkpoints exist and because a non-zero count is the
single most useful thing this script can tell you.

Note the layers are not independent: if layer 1 has even one unit that is
not IMPOSSIBLE, then `spk1` is non-zero on some timing-silent steps, so
layer 2 can fire on those steps regardless of its own numbers.

Usage
-----
    python check_bias_voltage.py                        # outputs/model/
    python check_bias_voltage.py --model_dir run7       # outputs/run7/model/
    python check_bias_voltage.py --top 20

Writes nothing; prints to stdout.
"""

import argparse
import os
from pathlib import Path

import numpy as np

# Per-unit state codes, in increasing order of badness.  Index into this
# list with the output of `classify`.
STATES = ["IMPOSSIBLE", "ASSIST", "SPONTANEOUS", "ALWAYS"]


# ─────────────────────────────────────────────────────────────────────
# Analysis.  Pure numpy -- no torch, no config parsing, no I/O, so it can
# be exercised on hand-built arrays.
# ─────────────────────────────────────────────────────────────────────

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def tau_of(decay):
    """Membrane time constant implied by a per-step decay factor."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(decay > 0, -1.0 / np.log(np.clip(decay, 1e-12, 1 - 1e-12)),
                        0.0)


def classify(th, tonic, decay):
    """
    th    : (G, H)          effective threshold, thresh - v
    tonic : (n_gaits, G, H) per-step drive on a timing-silent step
    decay : (G, H)          sigmoid(beta_logit)

    Returns (code, ceiling), both (n_gaits, G, H).  `ceiling` is the highest
    membrane the tonic drive alone can reach, i.e. tonic/(1-decay).
    """
    th = th[None]
    ceiling = np.where(tonic > 0, tonic / (1.0 - decay[None]), 0.0)

    code = np.zeros(tonic.shape, dtype=np.int8)          # IMPOSSIBLE
    code[tonic > 0] = 1                                  # ASSIST
    code[(tonic > 0) & (ceiling > th)] = 2               # SPONTANEOUS
    code[np.broadcast_to(th <= 0, tonic.shape)] = 3      # ALWAYS (dominates)
    return code, ceiling


def film_beta(film_weight, n_gaits, G, H):
    """
    Split a film embedding row into its beta half.

    `train.py` stores film as Embedding(max_gaits, 2*G*H) viewed as
    (2, G, H) with gamma first, beta second.  Only the first `n_gaits` rows
    ever receive gradient; the rest stay at their init (gamma 1, beta 0).
    """
    w = np.asarray(film_weight)[:n_gaits].reshape(n_gaits, 2, G, H)
    return w[:, 1]


# ─────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────

def report_layer(name, th, tonic, decay, gait_names, tonic_live, top):
    """One layer's block of output.  Returns True if any unit can fire."""
    G, H = th.shape
    print(f"\n─── {name} ─────────────────────────────────────────────")
    if not tonic_live:
        print("    film not applied to this layer (--sub_film), so the only")
        print("    silent-step drive would be zero; checking thresholds only.")
        tonic = np.zeros_like(tonic)

    print(f"    th = thresh - v : min {th.min():+.4f}  max {th.max():+.4f}"
          f"   (th <= 0 on {int((th <= 0).sum())} / {th.size} units)")
    print(f"    tau             : min {tau_of(decay).min():7.2f}  "
          f"max {tau_of(decay).max():7.2f} steps")
    if tonic_live:
        print(f"    film beta       : min {tonic.min():+.5f}  "
              f"max {tonic.max():+.5f}  ({(tonic > 0).mean() * 100:.1f}% > 0)")

    code, ceiling = classify(th, tonic, decay)

    print()
    print("    gait           " + "".join(f"{s:>13}" for s in STATES))
    for g, gname in enumerate(gait_names):
        counts = [int((code[g] == k).sum()) for k in range(len(STATES))]
        print(f"    {gname:<15}" + "".join(f"{c:>13,}" for c in counts))

    # Worst offenders, ranked by how far the tonic drive alone can push the
    # membrane past the unit's own threshold.  ALWAYS units get +inf so they
    # sort first: their threshold is non-positive, so a ratio against it is
    # meaningless and is printed as "always" rather than a number.
    firing = code > 0
    if firing.any():
        thb   = np.broadcast_to(th, code.shape)
        score = np.full(code.shape, -np.inf)
        rank  = (code == 1) | (code == 2)          # th > 0 guaranteed here
        score[rank] = (ceiling / thb)[rank]
        score[code == 3] = np.inf

        taus = tau_of(decay)
        print(f"\n    worst {min(top, int(firing.sum()))} units "
              f"(ceiling / th, >1 means it fires unaided):")
        for idx in np.argsort(score.ravel())[::-1][:top]:
            g, grp, u = np.unravel_index(idx, score.shape)
            if score[g, grp, u] == -np.inf:
                break
            ratio = ("     always" if code[g, grp, u] == 3
                     else f"{score[g, grp, u]:8.2f}x th")
            print(f"      {gait_names[g]:<10} group {grp:>3}  unit {u:>4}  "
                  f"th {th[grp, u]:+.4f}  beta {tonic[g, grp, u]:+.5f}  "
                  f"tau {taus[grp, u]:7.2f}  "
                  f"ceiling {ceiling[g, grp, u]:+.4f}  "
                  f"= {ratio}   [{STATES[code[g, grp, u]]}]")
    else:
        print("\n    no unit in this layer can fire without a timing spike.")

    return bool(firing.any())


# ─────────────────────────────────────────────────────────────────────
# Loading.  torch and train.py are imported here, not at module scope, so
# the analysis above stays usable without either.
# ─────────────────────────────────────────────────────────────────────

def load(model_dir_arg, ckpt, cfg_name, this_file_dir):
    import json

    import torch                                     # noqa: F401  (load only)
    from train import cfg_get, in_path, outputs_path

    model_dir = outputs_path(this_file_dir, model_dir_arg)
    cfg_path  = in_path(model_dir, cfg_name)
    ckpt_path = in_path(model_dir, ckpt)
    for p in (cfg_path, ckpt_path):
        if not p.exists():
            raise FileNotFoundError(f"Not found: {p}")

    cfg = json.loads(cfg_path.read_text())
    sd  = torch.load(ckpt_path, map_location="cpu")
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
    sd = {k: v.detach().float().cpu().numpy() for k, v in sd.items()}

    print(f"  Checkpoint {ckpt_path}")
    print(f"  Config     {cfg_path}  (version {cfg.get('config_version', '?')})")
    return cfg, sd, cfg_get


def main():
    this_file_dir = os.path.dirname(os.path.abspath(__file__))

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_dir", type=str, default="",
                    help="Run name under outputs/, e.g. 'run7'. Empty means "
                         "outputs/ itself, matching visualize_timing.py.")
    ap.add_argument("--ckpt", type=str, default="best_model.pt")
    ap.add_argument("--cfg",  type=str, default="cpg_lif_snn_config.json")
    ap.add_argument("--top",  type=int, default=10,
                    help="How many worst-offending units to list per layer.")
    args = ap.parse_args()

    cfg, sd, cfg_get = load(args.model_dir, args.ckpt, args.cfg, this_file_dir)

    arch      = cfg_get(cfg, "arch", "dense")
    bias_mode = cfg_get(cfg, "bias_mode", "current")
    gate_mode = cfg_get(cfg, "gate_mode", "none")
    sub_film  = cfg_get(cfg, "sub_film", "both")
    thresh    = float(cfg_get(cfg, "thresh", 1.0))
    n_gaits   = int(cfg_get(cfg, "n_gaits", 1))
    names     = cfg_get(cfg, "gait_names") or [f"g{i}" for i in range(n_gaits)]
    names     = list(names)[:n_gaits]

    print(f"  arch={arch}  bias_mode={bias_mode}  gate_mode={gate_mode}  "
          f"sub_film={sub_film}  thresh={thresh}  n_gaits={n_gaits}")

    if arch != "timing_grouped":
        print("\n  Nothing to check: no timing layer in this architecture.")
        return
    if bias_mode != "voltage":
        print(f"\n  Nothing to check: bias_mode={bias_mode!r} has no v1/v2. "
              f"In 'current' mode b1/b2 are a per-step drive by construction, "
              f"so hidden spikes on timing-silent steps are expected, not a "
              f"leak.")
        return
    if gate_mode != "none":
        print(f"\n  NOTE gate_mode={gate_mode!r}: spk1/spk2 are multiplied by "
              f"the gate, so a leak found below is suppressed at the layer "
              f"OUTPUT even though the membrane still crosses threshold. Run "
              f"--gate_mode none to see it in the spike trains.")

    G, Hg = sd["v1"].shape
    print(f"  G={G} groups, Hg={Hg} units per group\n")

    any_leak = False
    for layer, vkey, bkey, fkey, flag in (
            ("sub-net layer 1", "v1", "beta1_logit", "film1", "l1"),
            ("sub-net layer 2", "v2", "beta2_logit", "film2", "l2")):
        live  = sub_film in (flag, "both")
        tonic = film_beta(sd[f"{fkey}.weight"], n_gaits, G, Hg)
        any_leak |= report_layer(layer, thresh - sd[vkey], tonic,
                                 sigmoid(sd[bkey]), names, live, args.top)

    # The timing layer itself: no film, so its only failure mode is th <= 0.
    th_t = thresh - np.asarray(sd["v_t.weight"])[:n_gaits]        # (n_gaits, G)
    print("\n─── timing layer ──────────────────────────────────────")
    print(f"    th = thresh - v_t : min {th_t.min():+.4f}  "
          f"max {th_t.max():+.4f}   (th <= 0 on {int((th_t <= 0).sum())} / "
          f"{th_t.size} gait-neuron pairs)")
    print("    no film on this layer, so th <= 0 is the only way it fires "
          "without CPG input.")

    print("\n═══ verdict ═══════════════════════════════════════════")
    if any_leak:
        print("  A hidden unit CAN cross threshold on a timing-silent step.")
        print("  Look at the state column above to see which mechanism.")
    else:
        print("  No hidden unit can cross threshold without a timing spike.")
        print("  If a run still shows hidden spikes on silent steps, the cause")
        print("  is NOT the voltage bias or film beta -- look elsewhere.")


if __name__ == "__main__":
    main()