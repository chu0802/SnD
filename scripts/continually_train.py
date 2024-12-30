import argparse
from collections import deque
from copy import deepcopy
from pathlib import Path

from scripts.utils import DEFAULT_DATASET_SEQ, ContinualTrainer


def parse_dataset_seq(args):
    if args.dataset_seq is None:
        dataset_seq = deque(deepcopy(DEFAULT_DATASET_SEQ))
        dataset_seq.rotate(args.order)
        sub_output_dir = f"order_{args.order}"
    else:
        dataset_seq = args.dataset_seq.split(",")
        sub_output_dir = args.sub_output_dir
    return dataset_seq, sub_output_dir


def main(args):
    dataset_seq, sub_output_dir = parse_dataset_seq(args)

    continual_trainer = ContinualTrainer(
        config_path=args.config_path,
        module=args.module,
        training_dataset_seq=dataset_seq,
        sub_output_dir=sub_output_dir,
        output_root=args.output_root,
        max_epoch=args.max_epoch,
        max_iterations=args.max_iterations,
        distributed=args.distributed,
        nnodes=args.nnodes,
        nproc_per_node=args.nproc_per_node,
        method_config=args.method_config,
    )

    continual_trainer.train_and_eval(args.pretrained_dataset)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config_path", type=str, default="configs/split_teacher_config.yaml"
    )
    p.add_argument(
        "--dataset_seq",
        type=str,
        default=None,
        help="the sequence of training datasets, splitted by comma",
    )
    p.add_argument("--module", type=str, default="main.train")
    p.add_argument("--pretrained_dataset", type=str, default=None)
    p.add_argument("--distributed", action="store_true")
    p.add_argument("--nnodes", type=int, default=1)
    p.add_argument("--nproc_per_node", type=int, default=1)
    p.add_argument("--max_epoch", type=int, default=10)
    p.add_argument("--max_iterations", type=int, default=1000)
    p.add_argument("--order", type=int, default=0)
    p.add_argument("--output_root", type=Path, default=Path("outputs"))
    p.add_argument("--sub_output_dir", type=str, default="default")
    p.add_argument("--method_config", nargs="+")
    args = p.parse_args()

    if args.method_config is not None:
        args.method_config = {
            k.split("=")[0]: k.split("=")[1] for k in args.method_config
        }

    main(args)
