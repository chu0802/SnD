import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from scripts.metrics import DEFAULT_ZERO_SHOT_PERFORMANCE
from scripts.utils import DEFAULT_DATASET_SEQ

plt.rcParams["font.family"] = "Times New Roman"

METHOD_MAP = {
    "Continual-FT": "base",
    "LwF": "lwf",
    "iCaRL": "icarl",
    "ZSCL": "zscl",
}

VISUALIZED_DATASET_NAME_MAP = {
    "fgvc-aircraft": "Aircraft",
    "dtd": "DTD",
    "eurosat": "EuroSAT",
    "flowers-102": "Flowers",
    "food-101": "Food",
    "oxford-pets": "Pets",
    "stanford-cars": "Cars",
    "ucf-101": "UCF101",
}


def parse_results(method="split_teacher_pure_clip", is_mdcil=False):
    config_name = f"{method}_config"

    res_list = []
    for order in range(8):
        res_path = (
            Path("/work/chu980802/mix-teacher")
            / method
            / "outputs"
            / f"order_{order}"
            / config_name
            / "final_results.json"
        )
        with res_path.open("r") as f:
            res = json.load(f)
        res_list.append(pd.DataFrame(res).T)
    return res_list


def plot_figure(
    data_dict, zero_shot, title, legend=None, save_path="comparison_plot.pdf"
):
    plt.figure(figsize=(10, 10))
    markers = ["o", "s", "D", "^", "P"]  # Different markers for each series
    # colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    colors = ["#92CD00", "#A3D1F2", "#F4B6C2", "#FED1BD", "#957DAD", "#88D8C0"]

    plt.scatter(
        0, zero_shot, marker="*", s=800, color="#88D8C0", label="Zero-shot", zorder=3
    )

    for (label, data), marker, color in zip(data_dict.items(), markers, colors):
        plt.plot(
            range(len(data) + 1),
            [zero_shot] + data,
            label=label,
            marker=marker,
            lw=3,
            markersize=10,
            color=color,
        )
    plt.title(title, fontsize=40)
    plt.xlabel("Task sequence", fontsize=40)
    plt.ylabel("Accuracy", fontsize=40)
    plt.tick_params(labelsize=30)
    plt.xticks(range(9), range(0, 9))  # Example x-axis labels
    if legend is not None:
        plt.legend(fontsize=26, loc=legend)
    plt.grid(linestyle=":")
    plt.tight_layout()
    # Save the figure as a PDF
    plt.savefig(save_path)


res_list_dict = {k: parse_results(v) for k, v in METHOD_MAP.items()}

res_list = [
    {method: res[order] for method, res in res_list_dict.items()} for order in range(8)
]

# Catastrophic forgetting
for order in range(8):
    dataset_name = {res.index[0] for res in res_list[order].values()}
    assert len(dataset_name) == 1
    dataset_name = dataset_name.pop()
    display_order = DEFAULT_DATASET_SEQ.index(dataset_name)
    legend = "lower left" if order == 0 else None
    plot_figure(
        {
            method: res.loc[:, dataset_name].values.tolist()
            for method, res in res_list[order].items()
        },
        zero_shot=DEFAULT_ZERO_SHOT_PERFORMANCE[dataset_name],
        title="Acc. of the 1st task in $\mathcal{S}^<ORDER>$ (<DATASET>)".replace(
            "<ORDER>", str(display_order + 1)
        ).replace("<DATASET>", VISUALIZED_DATASET_NAME_MAP[dataset_name]),
        legend=legend,
        save_path=f"visualization/{dataset_name}_forgetting_order_{display_order+1}.pdf",
    )

# Zero-shot degradation
for order in range(8):
    dataset_name = {res.index[-1] for res in res_list[order].values()}
    assert len(dataset_name) == 1
    dataset_name = dataset_name.pop()
    display_order = (
        DEFAULT_DATASET_SEQ.index(dataset_name)
        + 1
        - 8 * (DEFAULT_DATASET_SEQ.index(dataset_name) == 7)
    )
    legend = "upper left" if order == 0 else None
    plot_figure(
        {
            method: res.loc[:, dataset_name].values.tolist()
            for method, res in res_list[order].items()
        },
        zero_shot=DEFAULT_ZERO_SHOT_PERFORMANCE[dataset_name],
        title="Acc. of the 8th task in $\mathcal{S}^<ORDER>$ (<DATASET>)".replace(
            "<ORDER>", str(display_order + 1)
        ).replace("<DATASET>", VISUALIZED_DATASET_NAME_MAP[dataset_name]),
        legend=legend,
        save_path=f"visualization/{dataset_name}_degradation_order_{display_order+1}.pdf",
    )
