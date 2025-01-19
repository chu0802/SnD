import argparse
from collections import deque
from copy import deepcopy
from pathlib import Path

from scripts.utils import DEFAULT_DATASET_SEQ, ContinualTrainer


def parse_dataset_seq(args):
    dataset_seq = deque(deepcopy(DEFAULT_DATASET_SEQ))
    dataset_seq.rotate(len(DEFAULT_DATASET_SEQ) - args.order)
    sub_output_dir = f"order_{args.order}"

    return dataset_seq, sub_output_dir


def main(args):
    dataset_seq, sub_output_dir = parse_dataset_seq(args)

    continual_trainer = ContinualTrainer(
        config_path=args.config_path,
        training_dataset_seq=dataset_seq,
        sub_output_dir=sub_output_dir,
        output_root=args.output_root,
        max_iterations=args.max_iterations,
        distributed=args.distributed,
        nproc_per_node=args.nproc_per_node,
        method_config=args.method_config,
    )

    continual_trainer.train_and_eval()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config_path", type=str, default="configs/snd_config_4_gpus.yaml"
    )

    p.add_argument("--max_iterations", type=int, default=1000)

    p.add_argument("--order", type=int, default=0)
    p.add_argument("--output_root", type=Path, default=Path("outputs"))

    p.add_argument("--distributed", action="store_true")
    p.add_argument("--nproc_per_node", type=int, default=1)

    p.add_argument("--method_config", nargs="+")
    args = p.parse_args()

    if args.method_config is not None:
        args.method_config = {
            k.split("=")[0]: k.split("=")[1] for k in args.method_config
        }

    main(args)
