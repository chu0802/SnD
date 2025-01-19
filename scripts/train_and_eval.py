import argparse

from scripts.utils import train_and_eval_script


def main(args):
    train_and_eval_script(
        config_path=args.config_path,
        training_dataset=args.dataset,
        pretrained_dataset=args.pretrained_dataset,
        max_iterations=args.max_iterations,
        sub_output_dir=args.sub_output_dir,
        timestamp=args.timestamp,
        distributed=args.distributed,
        nproc_per_node=args.nproc_per_node,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config_path", type=str, default="configs/snd_config_4_gpus.yaml", help="select the config file"
    )
    p.add_argument(
        "--pretrained_dataset",
        type=str,
        default=None,
        help="the latest training dataset, this is specified to choose the corresponding model. Do not set it if you want to select the original pre-trained CLIP model.",
    )
    p.add_argument(
        "--dataset", type=str, default="fgvc-aircraft", help="the dataset to train"
    )
    p.add_argument(
        "--sub_output_dir",
        type=str,
        default="default",
        help="the sub-directory to save the training results, choose any name you want",
    )

    p.add_argument("--max_iterations", type=int, default=1000, help="the maximum number of iterations to train.")
    p.add_argument(
        "--timestamp", 
        type=str, 
        default="latest", 
        help="select the timestamp folder. This is used to select the model you fine-tuned previously. Do not set it if you want to select the latest model.",
    )

    p.add_argument("--distributed", action="store_true", help="use distributed training")
    p.add_argument("--nproc_per_node", type=int, default=1, help="number of GPUs per node")

    args = p.parse_args()

    main(args)
