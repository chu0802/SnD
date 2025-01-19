import argparse
from pathlib import Path

from scripts.utils import (
    DEFAULT_DATASET_SEQ,
    eval_on_multiple_datasets_script,
)


def main(args):
    args.dataset_seq = (
        DEFAULT_DATASET_SEQ if args.dataset_seq is None else args.dataset_seq.split(",")
    )

    eval_on_multiple_datasets_script(
        datasets=args.dataset_seq,
        pretrained_model_path=args.model_path,
        dump_result_path=args.model_path.parent / "eval_results.json",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset_seq",
        type=str,
        default=None,
        help="the sequence of evaluation datasets, splitted by comma. Do not set it if you want to evaluate on all of our default datasets",
    )
    p.add_argument(
        "--model_path",
        type=Path,
        required=True,
        help="specify the path of the model to evaluate",
    )
    args = p.parse_args()

    main(args)
