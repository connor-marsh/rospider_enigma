from matplotlib import pyplot as plt
import numpy as np

CPG_FROM_FB_WEIGHT = -2_000_0.0

CPG_W_BY_N = {
    3: np.asarray([
        [    0.0      , -523.65135942, -593.28982051],
        [-696.81822016,     0.0      , -632.34680962],
        [-687.56816569, -577.5693762 ,     0.0      ],
    ], dtype=np.float64),

    4: np.asarray([
        [    0.0      , -648.52905924, -449.60304695, -413.48426163],
        [-369.91504928,     0.0      , -592.29635234, -568.0712858 ],
        [-412.08729881, -391.54918498,     0.0      , -618.03381552],
        [-498.16458351, -655.01105883, -345.38277449,     0.0      ],
    ], dtype=np.float64),

    6: np.asarray([
        [    0.0      , -375.86210512, -518.18703523, -371.82375498, -399.74231244, -487.45119873],
        [-531.99480471,     0.0      , -489.1139223 , -128.33470562, -404.33117771, -628.03347932],
        [-529.89653583, -418.34662835,     0.0      , -543.37143674, -336.83773596, -679.12224243],
        [-674.09562904, -130.56007131, -297.35360394,     0.0      , -363.1208234 , -425.10847629],
        [-486.03391005, -386.7920052 , -412.91478912, -437.7646991 ,     0.0      , -288.47748806],
        [-112.97808475, -510.59115452, -367.63412082, -374.83106147, -393.86103887,     0.0      ],
    ], dtype=np.float64),
}


def cpg_weight_matrix(N):
    """Coupling matrix for an N-neuron CPG.  Raises rather than falling back,
    so a typo'd --n_cpg_neurons fails at startup instead of silently
    training against the wrong oscillator."""
    if N not in CPG_W_BY_N:
        raise ValueError(
            f"No CPG weight matrix for N={N}; available: "
            f"{sorted(CPG_W_BY_N)}.  Add one to CPG_W_BY_N.")
    return (CPG_W_BY_N[N].copy()*0.1)

class LIFGeneralArray:
    """Vectorised current-based LIF with 2-stage filtering and refractoriness."""

    def __init__(self, num_neurons, vth, du, dv, bias=0.0, u=0.0, v=0.0,
                 refractory_period=0):
        self.vth  = vth
        self.du   = du
        self.dv   = dv
        self.bias = bias
        self.u    = np.full(num_neurons, float(u))
        self.v    = np.full(num_neurons, float(v))
        self.refractory_period    = refractory_period
        self.time_since_last_spike = np.zeros(num_neurons)
        self.num_neurons = num_neurons

    def next_step(self, current):
        self.u = self.u * (1 - self.du) + current
        self.v = self.v * (1 - self.dv) + self.u + self.bias

        refractory_mask = self.time_since_last_spike > 0
        self.v[refractory_mask] = 0
        self.time_since_last_spike = np.clip(
            self.time_since_last_spike - 1, 0, None)

        spike = self.v >= self.vth
        self.v[spike] = 0
        self.time_since_last_spike[spike] = self.refractory_period
        return spike.astype(np.float32)

    def reset(self, u=0.0, v=0.0):
        self.u.fill(u)
        self.v.fill(v)
        self.time_since_last_spike.fill(0)


class BurstingLIF:
    """
    Main neuron + fast feedback neuron.

    The feedback neuron integrates `to_fb_weight` per main spike with no
    leak (dv_fb=0), so after ~vth_fb/to_fb_weight main spikes it fires and
    dumps `from_fb_weight` (large negative) into the main neuron, killing
    the burst.  With the supplied params that is 100/10 = 10 spikes/burst.
    """

    def __init__(self, num_neurons, vth_main, du_main, dv_main, refrac_main,
                 vth_fb, du_fb, dv_fb, refrac_fb, from_fb_weight, to_fb_weight):
        self.n_main = LIFGeneralArray(num_neurons, vth_main, du_main, dv_main,
                                      refractory_period=refrac_main)
        self.n_fb   = LIFGeneralArray(num_neurons, vth_fb, du_fb, dv_fb,
                                      refractory_period=refrac_fb)
        self.input_2_feedback_neuron_weight = to_fb_weight
        self.feedback_2_input_neuron_weight = from_fb_weight
        self.fb_current = np.zeros(num_neurons)

    def forward(self, current):
        main_spike = self.n_main.next_step(current + self.fb_current)
        fb_spike   = self.n_fb.next_step(
            main_spike * self.input_2_feedback_neuron_weight)
        self.fb_current = fb_spike * self.feedback_2_input_neuron_weight
        return main_spike, fb_spike

    def reset(self):
        self.n_main.reset()
        self.n_fb.reset()
        self.fb_current.fill(0.0)


class LIFCPGStepper:
    """
    Canonical spike generator — identical object is used for training data
    generation and for deployment on the Raspberry Pi.

    One `step()` advances exactly one timestep and returns the (N,) binary
    spike vector.  There is no integrator state that differs between batch
    and streaming use, so there is no train/deploy mismatch to reason about.
    """

    def __init__(self, N, W=None, i_app=8.0,
                 vth_main=100.0, du_main=0.1, dv_main=0.3, refrac_main=1,
                 vth_fb=100.0, du_fb=1.0, dv_fb=0.0, refrac_fb=1,
                 from_fb_weight=CPG_FROM_FB_WEIGHT, to_fb_weight=10.0):
        self.N     = int(N)
        self.W     = (cpg_weight_matrix(self.N) if W is None
                      else np.asarray(W, dtype=np.float64))
        if self.W.shape != (self.N, self.N):
            raise ValueError(
                f"CPG weight matrix is {self.W.shape}, expected "
                f"({self.N}, {self.N}).")
        self.i_app = float(i_app)
        self.core  = BurstingLIF(N, vth_main, du_main, dv_main, refrac_main,
                                 vth_fb, du_fb, dv_fb, refrac_fb,
                                 from_fb_weight, to_fb_weight)
        self.inter_neuron_current = np.zeros(N)
        self.t = 0

    def step(self):
        spk = self.core.forward(self.inter_neuron_current + self.i_app)[0]
        self.inter_neuron_current = self.W @ spk
        self.t += 1
        return spk

    def step_chunk(self, n_steps):
        out = np.zeros((n_steps, self.N), dtype=np.float32)
        for k in range(n_steps):
            out[k] = self.step()
        return out

    def reset(self):
        self.core.reset()
        self.inter_neuron_current.fill(0.0)
        self.t = 0


def run_cpg(N, tmax=120_000, warmup=2_000, i_app=8.0):
    """Warm up, then collect the spike train used for training.

    `N` is deliberately positional-with-no-default: it selects the coupling
    matrix, and a wrong value changes the oscillator rather than raising, so
    the caller is made to say it.

    from_fb_weight is not a parameter here: it is fixed at
    CPG_FROM_FB_WEIGHT (see LIFCPGStepper) for every N, so there is nothing
    for a caller to get wrong by omission.
    """
    cpg = LIFCPGStepper(N=N, i_app=i_app)
    print(f"  N={N}  i_app={i_app}  from_fb_weight={CPG_FROM_FB_WEIGHT:g}")
    print(f"  Warming up CPG ({warmup} steps) ...")
    cpg.step_chunk(warmup)
    print(f"  Collecting {tmax} steps ...")
    spikes = cpg.step_chunk(tmax)

    counts = spikes.sum(0).astype(int)
    print(f"  Spikes per neuron : {counts.tolist()}")
    if counts.min() == 0:
        raise RuntimeError("A CPG neuron never fired — check W / i_app.")
    return spikes


if __name__ == "__main__":
    spikes = run_cpg(N=6, tmax=10000, warmup=2000,
                         i_app=8.0)
    spikes = spikes.T
    print(spikes.shape)
    n_show=1000
    fig, axes = plt.subplots(1, 1)
    for i in range(spikes.shape[0]):
        axes.plot(spikes[i, :n_show], label=f"Neuron {i+1} spikes")
    axes.legend()


    plt.suptitle("BLIF CPG Spikes", fontsize=12)
    plt.tight_layout()
    plt.show()