"""
Visualize a trained CPG-SNN run
===============================
Diagnostic figures and alignment scores by default; publication figures with --publication.
Loads a checkpoint (.pt) and its config (.json) written by train.py and replays the CPG from the
config's own parameters. The flows live in cpgsnn/viz_pipeline.py, the figures in
cpgsnn/plotting.py.

Usage
-----
    python visualize.py                       # reads outputs/, writes into it
    python visualize.py --model_dir outputs/run7
    python visualize.py --ckpt epoch100.pt    # another snapshot -> viz_epoch100/
    python visualize.py --scores_only         # alignment scores only, no figures
    python visualize.py --drive_metric spikes # cpg_lif: exact spike counts
    python visualize.py --gaits wkF bk         # subset
    python visualize.py --n_cycles 4 --no_pred # tighter / faster

Publication mode
----------------
    python visualize.py --model_dir test1 --publication
    python visualize.py --model_dir test1 --publication --pub_cycles 4
"""
import os

from cpgsnn.common import outputs_path
from cpgsnn.viz_pipeline import build_parser, run_publication, run_visualization

this_file_dir = os.path.dirname(os.path.abspath(__file__))


def main():
    args = build_parser().parse_args()
    model_dir = outputs_path(this_file_dir, args.model_dir)
    out_dir = None if args.out_dir is None else outputs_path(this_file_dir, args.out_dir)
    (run_publication if args.publication else run_visualization)(model_dir, out_dir, args)


if __name__ == "__main__":
    main()
