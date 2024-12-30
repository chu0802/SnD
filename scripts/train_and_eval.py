import argparse

from scripts.utils import train_and_eval_script


def main(args):
    train_and_eval_script(
        config_path=args.config_path,
        training_dataset=args.dataset,
        pretrained_dataset=args.pretrained_dataset,
        sample_num=args.sample_num,
        max_epoch=args.train_epoch,
        max_iterations=args.max_iterations,
        sub_output_dir=args.sub_output_dir,
        eval_epoch=args.eval_epoch,
        timestamp=args.timestamp,
        distributed=args.distributed,
        nnodes=args.nnodes,
        nproc_per_node=args.nproc_per_node,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config_path", type=str, default="configs/split_teacher_config.yaml"
    )
    p.add_argument("--pretrained_dataset", type=str, default=None)
    p.add_argument("--dataset", type=str, default="fgvc-aircraft")
    p.add_argument("--sub_output_dir", type=str, default="default")
    p.add_argument(
        "--sample_num",
        type=int,
        default=-1,
        help="sample number for training, if sample_num is -1, use all samples. this is usually used for debugging.",
    )
    p.add_argument("--train_epoch", type=int, default=10)
    p.add_argument("--max_iterations", type=int, default=1000)
    p.add_argument(
        "--timestamp", type=str, default="latest", help="select the timestamp folder"
    )
    p.add_argument(
        "--eval_epoch",
        type=str,
        default="latest",
        help="determine to use the model saved in which epoches to evaluate",
    )
    p.add_argument("--distributed", action="store_true")
    p.add_argument("--nnodes", type=int, default=1)
    p.add_argument("--nproc_per_node", type=int, default=1)

    args = p.parse_args()

    main(args)
