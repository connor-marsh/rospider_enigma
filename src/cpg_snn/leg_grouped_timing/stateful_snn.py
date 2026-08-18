"""
stateful_snn.py
===============
Stateful version of `cpg_utils.CPG_SNN`.

`CPG_SNN` zeroes all four membranes, runs `seq_len` events, and reads the
output at the last one.  `CPG_SNN_Stateful` has the same layers and the same
parameter names, but takes ONE event per call and keeps its membranes
between calls, so it produces one prediction per event.

Because the module tree is identical, one `best_model.pt` loads into either
class with `strict=True`.
"""

from pathlib import Path

import torch
import torch.nn as nn

import snntorch as snn
from snntorch import surrogate


class CPG_SNN_Stateful(nn.Module):
    """
    Step-at-a-time twin of `cpg_utils.CPG_SNN`.

    State is a dict of four (B, hidden) membranes so it is easy to inspect
    or reset mid-run:  m1, m2, m3 (hidden LIFs), mo (analog readout).
    """

    def __init__(self, n_in, hidden=128, n_out=8, n_gaits=4,
                 beta=0.9, spike_grad=None):
        super().__init__()
        spike_grad = spike_grad or surrogate.fast_sigmoid(slope=25)

        # Layer names and order must match CPG_SNN exactly.
        self.fc1     = nn.Linear(n_in, hidden)
        self.ln1     = nn.LayerNorm(hidden)
        self.lif1    = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc2     = nn.Linear(hidden, hidden)
        self.ln2     = nn.LayerNorm(hidden)
        self.lif2    = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc3     = nn.Linear(hidden, hidden)
        self.ln3     = nn.LayerNorm(hidden)
        self.lif3    = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc_read = nn.Linear(hidden, hidden)
        self.lif_out = snn.Leaky(beta=beta, spike_grad=spike_grad,
                                 threshold=1e9)
        self.fc_out  = nn.Linear(hidden, n_out)

        self.n_in   = n_in
        self.hidden = hidden
        self.n_out  = n_out

    def init_state(self, batch=1, device=None, dtype=torch.float32):
        device = device or next(self.parameters()).device
        z = lambda: torch.zeros(batch, self.hidden, device=device, dtype=dtype)
        return {"m1": z(), "m2": z(), "m3": z(), "mo": z()}

    def step(self, x_t, state):
        """
        x_t   : (B, n_in)  one event's features
        state : dict from init_state()
        returns (out, state)  with out : (B, n_out)
        """
        s1, m1 = self.lif1(self.ln1(self.fc1(x_t)),  state["m1"])
        s2, m2 = self.lif2(self.ln2(self.fc2(s1)),   state["m2"])
        s3, m3 = self.lif3(self.ln3(self.fc3(s2)),   state["m3"])
        _,  mo = self.lif_out(self.fc_read(s3),      state["mo"])
        out = self.fc_out(mo)
        return out, {"m1": m1, "m2": m2, "m3": m3, "mo": mo}

    def forward(self, x_seq, state=None):
        """
        x_seq : (T, B, n_in) contiguous event stream (NOT a batch of windows).
        Returns (T, B, n_out): one prediction per event, state carried.
        """
        if state is None:
            state = self.init_state(x_seq.shape[1], x_seq.device, x_seq.dtype)
        outs = []
        for t in range(x_seq.shape[0]):
            out, state = self.step(x_seq[t], state)
            outs.append(out)
        return torch.stack(outs), state


# ═══════════════════════════════════════════════════════════════════
# Checkpoint helpers
# ═══════════════════════════════════════════════════════════════════

def load_checkpoint(ckpt_path):
    """
    Read `best_model.pt` and return (state_dict, arch).

    Handles the `_orig_mod.` prefix that `torch.compile` adds when training
    on CUDA, and recovers n_in / hidden / n_out / beta from the tensors —
    `cpg_snn_config.json` does not store hidden or beta.
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "fc1.weight" not in sd:
        for key in ("state_dict", "model", "model_state_dict"):
            if isinstance(sd.get(key), dict):
                sd = sd[key]
                break
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}

    if "fc1.weight" not in sd or "fc_out.weight" not in sd:
        raise KeyError(f"Not a CPG_SNN checkpoint. Keys: {sorted(sd)[:8]} ...")

    hidden, n_in = sd["fc1.weight"].shape
    arch = {"n_in": int(n_in), "hidden": int(hidden),
            "n_out": int(sd["fc_out.weight"].shape[0]), "beta": None}
    for key in ("lif1.beta", "lif2.beta", "lif3.beta"):
        if key in sd:
            arch["beta"] = float(sd[key].reshape(-1)[0])
            break
    return sd, arch