import argparse

from scripts.utils import (
    DEFAULT_DATASET_SEQ,
    DEFAULT_OUTPUT_ROOT,
    eval_on_multiple_datasets_script,
    get_model_path,
    get_output_dataset_dir,
)


def main(args):
    args.dataset_seq = (
        DEFAULT_DATASET_SEQ if args.dataset_seq is None else args.dataset_seq.split(",")
    )
    output_root = DEFAULT_OUTPUT_ROOT / args.sub_output_dir

    model_path = get_model_path(
        args.pretrained_dataset, output_root=output_root, epoch=args.eval_epoch
    )

    eval_results_path = (
        get_output_dataset_dir(args.pretrained_dataset, output_root=output_root)
        / "eval_results.json"
    )

    eval_on_multiple_datasets_script(
        datasets=args.dataset_seq,
        pretrained_model_path=model_path,
        dump_result_path=eval_results_path,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset_seq",
        type=str,
        default=None,
        help="the sequence of training datasets, splitted by comma",
    )
    p.add_argument("--pretrained_dataset", type=str, default=None)
    p.add_argument("--sub_output_dir", type=str, default="default")
    p.add_argument(
        "--eval_epoch",
        type=str,
        default="latest",
        help="determine to use the model saved in which epoches to evaluate",
    )
    args = p.parse_args()

    main(args)
