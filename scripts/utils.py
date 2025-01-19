import json
import subprocess
from ast import literal_eval
from pathlib import Path
from typing import List, Union

DEFAULT_OUTPUT_ROOT = Path("outputs")
DEFAULT_STORAGE_ROOT = Path("/work/chu980802/mix-teacher")

DEFAULT_DATASET_SEQ = [
    "fgvc-aircraft",
    "dtd",
    "eurosat",
    "flowers-102",
    "food-101",
    "oxford-pets",
    "stanford-cars",
    "ucf-101",
]


class ContinualTrainer:
    def __init__(
        self,
        config_path: str = "configs/mix_teacher_config.yaml",
        module: str = "main.train",
        training_dataset_seq: List[str] = DEFAULT_DATASET_SEQ,
        eval_dataset_seq: List[str] = None,
        output_root: Path = DEFAULT_OUTPUT_ROOT,
        sub_output_dir: str = "default",
        method_config=None,
        max_epoch: int = 10,
        max_iterations: int = 1000,
        distributed: bool = False,
        nnodes: int = 1,
        nproc_per_node: int = 1,
    ):
        self.config_path = config_path
        self.module = module
        self.training_dataset_seq = training_dataset_seq
        self.eval_dataset_seq = (
            training_dataset_seq if eval_dataset_seq is None else eval_dataset_seq
        )
        self.train_eval_config = {
            "distributed": distributed,
            "nnodes": nnodes,
            "nproc_per_node": nproc_per_node,
            "max_epoch": max_epoch,
            "max_iterations": max_iterations,
        }

        self.output_root = output_root
        self.sub_output_dir = sub_output_dir

        self.method_config = method_config if method_config is not None else {}

        self.output_dir = (
            self.output_root / self.sub_output_dir / Path(self.config_path).stem
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def aggregate_results(self, training_dataset_seq, ouptut_root):
        results_dict = dict()
        for dataset in training_dataset_seq:
            eval_result_path = (
                get_output_dataset_dir(dataset, output_root=ouptut_root)
                / "eval_results.json"
            )

            with eval_result_path.open("r") as f:
                results = json.load(f)

            results_dict[dataset] = results

        return results_dict

    @classmethod
    def format_results(
        self,
        res_dict,
        training_dataset_seq,
        eval_dataset_seq,
        pad=4,
        decimal=2,
    ):
        longest_training_dataset_name_len = max([len(k) for k in training_dataset_seq])
        lines = []
        lines.append(
            (" " * pad).join(
                [" " * longest_training_dataset_name_len]
                + [
                    f"%{max(len(dataset), 5)}s" % (dataset)
                    for dataset in eval_dataset_seq
                ]
            )
        )

        for training_dataset in training_dataset_seq:
            line = [f"%{longest_training_dataset_name_len}s" % (training_dataset)]
            line += [
                f"%{len(eval_dataset)}s"
                % (f"{100*res_dict[training_dataset][eval_dataset]:.{decimal}f}")
                for eval_dataset in eval_dataset_seq
            ]
            lines.append((" " * pad).join(line))

        return "\n".join(lines) + "\n"

    def train_and_eval(self, format=True):
        pretrained_dataset = None
        for training_dataset in self.training_dataset_seq:
            train_and_eval_script(
                config_path=self.config_path,
                training_module=self.module,
                training_dataset=training_dataset,
                pretrained_dataset=pretrained_dataset,
                eval_dataset_seq=self.eval_dataset_seq,
                output_root=self.output_root,
                sub_output_dir=self.sub_output_dir,
                **self.train_eval_config,
                **self.method_config,
            )
            pretrained_dataset = training_dataset

        res = self.aggregate_results(
            training_dataset_seq=self.training_dataset_seq,
            ouptut_root=self.output_dir.parent,
        )

        with (self.output_dir / "final_results.json").open("w") as f:
            json.dump(res, f, indent=4)

        if format:
            formatted_results = self.format_results(
                res, self.training_dataset_seq, self.eval_dataset_seq
            )
            with (self.output_dir / "formatted_results.txt").open("w") as f:
                f.write(formatted_results)
            print(formatted_results)

        return res


def get_output_dataset_dir(
    dataset=None, output_root=DEFAULT_OUTPUT_ROOT, timestamp="latest"
):
    if dataset is None:
        dataset = "openai"
    return output_root / dataset / timestamp


def get_model_path(
    dataset=None, output_root=DEFAULT_OUTPUT_ROOT, timestamp="latest", epoch="latest"
):
    if dataset is None:
        return "openai"
    model_dir = get_output_dataset_dir(dataset, output_root, timestamp)
    return model_dir / f"checkpoint_{epoch}.pth"


def start_subprocess(command, print_command=False):
    if isinstance(command, list):
        command = " ".join(command)
    if print_command:
        print(command + "\n")
    output = subprocess.check_output(command, shell=True)

    return output.decode("utf-8")


def train_and_eval_script(
    config_path: str = "configs/mix_teacher_config.yaml",
    training_module: str = "main.train",
    training_dataset: str = "fgvc-aircraft",
    pretrained_dataset: str = None,
    eval_dataset_seq: List[str] = DEFAULT_DATASET_SEQ,
    sample_num: int = -1,
    max_epoch: int = 10,
    max_iterations: int = 1000,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    sub_output_dir: str = "default",
    eval_epoch: Union[int, str] = "latest",
    timestamp="latest",
    distributed=False,
    nnodes=1,
    nproc_per_node=1,
    **method_config,
):
    output_dir = output_root / sub_output_dir
    pretrained_model_path = get_model_path(
        pretrained_dataset, output_root=output_dir, timestamp=timestamp
    )

    training_script(
        config_path=config_path,
        training_module=training_module,
        dataset=training_dataset,
        pretrained_model_path=pretrained_model_path,
        sample_num=sample_num,
        max_epoch=max_epoch,
        max_iterations=max_iterations,
        output_root=output_root,
        sub_output_dir=sub_output_dir,
        distributed=distributed,
        nnodes=nnodes,
        nproc_per_node=nproc_per_node,
        **method_config,
    )

    model_path = get_model_path(
        training_dataset, output_root=output_dir, epoch=eval_epoch
    )
    eval_results_path = (
        get_output_dataset_dir(training_dataset, output_root=output_dir)
        / "eval_results.json"
    )

    eval_on_multiple_datasets_script(
        datasets=eval_dataset_seq,
        pretrained_model_path=model_path,
        dump_result_path=eval_results_path,
    )


def training_script(
    config_path,
    training_module="main.train",
    dataset="fgvc-aircraft",
    pretrained_model_path="openai",
    sample_num=-1,
    max_epoch=10,
    max_iterations=1000,
    output_root=DEFAULT_OUTPUT_ROOT,
    sub_output_dir="default",
    distributed=False,
    nnodes=1,
    nproc_per_node=1,
    **method_config,
):
    runner = (
        "python"
        if not distributed
        else f"torchrun --nnodes={nnodes} --nproc_per_node={nproc_per_node}"
    )
    command = [
        runner,
        "-m",
        training_module,
        "--cfg-path",
        config_path,
        "--options",
        f"data.name={dataset}",
        f"model.pretrained={pretrained_model_path}",
        f"data.sample_num={sample_num}",
        f"task.max_epoch={max_epoch}",
        f"task.max_iterations={max_iterations}",
        f"task.output_dir={output_root}/{sub_output_dir}",
        f"task.distributed={distributed}",
    ]

    if len(method_config) > 0:
        command += [f"method.{k}={v}" for k, v in method_config.items()]

    start_subprocess(command, print_command=True)


def eval_on_multiple_datasets_script(
    config_path="configs/inference_config.yaml",
    eval_module="main.evaluate",
    datasets=DEFAULT_DATASET_SEQ,
    pretrained_model_path="openai",
    sample_num=-1,
    dump_result_path=None,
):
    eval_results = {}
    for eval_dataset in datasets:
        command = [
            "python",
            "-m",
            eval_module,
            "--cfg-path",
            config_path,
            "--options",
            f"model.pretrained={pretrained_model_path}",
            f"data.name={eval_dataset}",
            f"data.sample_num={sample_num}",
        ]

        res = start_subprocess(command, print_command=True)

        eval_results[eval_dataset] = float(literal_eval(res)["zero shot"]["test_acc"])

    if dump_result_path:
        with open(dump_result_path, "w") as f:
            json.dump(eval_results, f, indent=4)

    return eval_results
