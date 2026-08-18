import matplotlib
from matplotlib import pyplot as plt
import numpy as np

from train import (
    # CPG + phase
    CPG_W_BY_N, CPG_FROM_FB_WEIGHT, CPG_PALETTE,
    cpg_weight_matrix, run_cpg, analyse_cpg, cycle_phase, cpg_spike_stats,
    # gait tables + targets
    GAIT_FILES_BY_N, load_gait_tables, upsample_gait_tables, build_targets,
    default_leg_layout, outputs_path,
    # sampler + training utilities
    StreamSampler, masked_loss, detach_state, apply_reset,
    make_gait_weights, MetricsWriter,
    # model primitives
    spike_fn, init_beta_logit,
    # plots + misc
    plot_cpg_raster, plot_training_curves, plot_reconstruction,
    plot_transition, json_safe, git_info,
)

matplotlib.use("QtAgg")


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