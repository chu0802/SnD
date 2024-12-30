import argparse
import json

import numpy as np
import pandas as pd

from scripts.utils import DEFAULT_DATASET_SEQ, DEFAULT_STORAGE_ROOT

DEFAULT_ZERO_SHOT_PERFORMANCE = {
    "fgvc-aircraft": 0.2391,
    "dtd": 0.4439,
    "eurosat": 0.4222,
    "flowers-102": 0.6740,
    "food-101": 0.84,
    "oxford-pets": 0.8727,
    "stanford-cars": 0.6551,
    "ucf-101": 0.6426,
}

DEFAULT_FINE_TUNING_PERFORMANCE = {
    "fgvc-aircraft": 0.5413,
    "dtd": 0.7973,
    "eurosat": 0.9875,
    "flowers-102": 0.9834,
    "food-101": 0.8985,
    "oxford-pets": 0.9414,
    "stanford-cars": 0.8607,
    "ucf-101": 0.8866,
}

DEFAULT_MDCIL_ZERO_SHOT_PERFORMANCE = {
    "fgvc-aircraft": 0.2385,
    "dtd": 0.3670,
    "eurosat": 0.3079,
    "flowers-102": 0.6719,
    "food-101": 0.8360,
    "oxford-pets": 0.8708,
    "stanford-cars": 0.6527,
    "ucf-101": 0.6297,
}

DEFAULT_MDCIL_FINE_TUNING_PERFORMANCE = {
    "fgvc-aircraft": 0.5395,
    "dtd": 0.7494,
    "eurosat": 0.9842,
    "flowers-102": 0.9834,
    "food-101": 0.8957,
    "oxford-pets": 0.9340,
    "stanford-cars": 0.8607,
    "ucf-101": 0.8768,
}


def zero_shot_performance(is_mdcil=False):
    return (
        DEFAULT_MDCIL_ZERO_SHOT_PERFORMANCE
        if is_mdcil
        else DEFAULT_ZERO_SHOT_PERFORMANCE
    )


def metric_to_dataframe(metric, index_name, columns=DEFAULT_DATASET_SEQ):
    try:
        data_frame = pd.DataFrame(metric, index=[index_name]).loc[:, columns]
        return data_frame.round(2)
    except:
        return metric


def zscl_trasnfer(res):
    metric = {
        res.index[i]: 100 * (res.loc[:, res.index[i]].iloc[:i].mean())
        for i in range(1, len(res))
    }
    metric[res.index[0]] = -1
    df = metric_to_dataframe(metric, "transfer")

    df["avg"] = df.to_numpy()[df.to_numpy() > 0].mean()

    return df.round(2)


def zscl_average(res):
    metric = {
        res.index[i]: 100 * res.loc[:, res.index[i]].to_numpy().mean()
        for i in range(len(res))
    }

    df = metric_to_dataframe(metric, "avg")
    df["avg"] = df.to_numpy().mean()

    return df.round(2)


def zscl_last(res):
    metric = {
        res.index[i]: 100 * res.loc[:, res.index[i]].iloc[-1] for i in range(len(res))
    }
    df = metric_to_dataframe(metric, "last")
    df["avg"] = df.to_numpy().mean()

    return df.round(2)


def max_catastrophic_forgetting(res_list):
    metric = {
        res.index[0]: 100
        # * (res.loc[:, res.index[0]].max() - res.loc[:, res.index[0]].min())
        # res.index[0]: 100 * res.loc[:, res.index[0]].min()
        * np.array(
            [
                res.iloc[:-1].loc[:, dataset].max() - res.iloc[-1].loc[dataset]
                for dataset in res.index[:-1]
            ]
        ).mean()
        for res in res_list
    }
    # print(metric)
    return metric_to_dataframe(metric, "catastrophic forgetting")


def max_zero_shot_degradation(res_list, is_mdcil=False):
    metric = {
        # res.index[-1]: 100
        res.index[0]: 100
        * (
            np.array(
                [
                    zero_shot_performance(is_mdcil=is_mdcil)[dataset]
                    - res.loc[:dataset, dataset].min()
                    for dataset in res.index[1:]
                ]
            ).mean()
        )
        # * (zero_shot_performance(is_mdcil=is_mdcil)[res.index[-1]] - res.loc[:, res.index[-1]].min())
        # res.index[-1]: 100 * res.loc[:, res.index[-1]].min()
        for res in res_list
    }
    # print(metric)
    return metric_to_dataframe(metric, "zero-shot degradation")


def avg_final_performance(res_list):
    metric = {res.index[0]: 100 * res.iloc[-1].mean() for res in res_list}
    # metric = 100 * pd.concat(
    #     [res.iloc[-1][DEFAULT_DATASET_SEQ] for res in res_list], axis=1
    # ).mean(axis=1)
    # return metric.to_frame("avg. final performance").T
    return metric_to_dataframe(metric, "avg. final performance")


def parse_results(method="split_teacher_pure_clip", is_mdcil=False):
    config_prefix = method.replace("_mdcil", "") if is_mdcil else method
    config_name = f"{config_prefix}_config"

    res_list = []
    output_dir = DEFAULT_STORAGE_ROOT / method / "outputs"
    num_orders = len(list(output_dir.iterdir()))

    for order in range(num_orders):
        res_path = (
            DEFAULT_STORAGE_ROOT
            / method
            / "outputs"
            / f"order_{order}"
            / config_name
            / "final_results.json"
        )
        with res_path.open("r") as f:
            res = pd.DataFrame(json.load(f)).T
        if res.iloc[0, 0] > 1:
            res *= 0.01
        res_list.append(res)
    return res_list


def main(args):
    res_list = parse_results(method=args.method, is_mdcil=args.is_mdcil)

    if args.order == "overall":
        forget = max_catastrophic_forgetting(res_list)
        degradation = max_zero_shot_degradation(res_list, is_mdcil=args.is_mdcil)
        avg = avg_final_performance(res_list)

        # print(forget, degradation, avg)
        print(pd.concat([forget, degradation, avg], axis=0).round(2))
    else:
        order = int(args.order)
        if args.zscl:
            transfer = zscl_trasnfer(res_list[order])
            avg = zscl_average(res_list[order])
            last = zscl_last(res_list[order])
            print(pd.concat([transfer, avg, last], axis=0).round(2))
        else:
            print((100 * res_list[order]).round(4).loc[:, res_list[order].index])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="split_teacher_pure_clip")
    p.add_argument(
        "--order",
        type=str,
        default="overall",
        choices=[str(i) for i in range(8)] + ["overall"],
    )
    p.add_argument(
        "--zscl",
        action="store_true",
        default=False,
        help="use metrics proposed by zscl",
    )
    args = p.parse_args()

    args.method = args.method.replace("ours", "split_teacher_pure_clip")

    args.is_mdcil = "mdcil" in args.method

    main(args)
