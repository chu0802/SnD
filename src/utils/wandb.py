import shutil

import wandb
from src.utils import is_main_process
from src.utils.config import flatten_config


def print_text_in_center_with_border(text, symbol="="):
    terminal_width, _ = shutil.get_terminal_size()
    padding = (terminal_width - len(text)) // 2

    print(symbol * padding + text + symbol * padding)


def wandb_logger(func):
    def wrap(config):
        if is_main_process():
            wandb.init(
                project=config.data.name,
                name=config.data.name,
                config=flatten_config(config),
            )
            func(config)
            wandb.finish()
        else:
            func(config)

    return wrap


def local_logger(func):
    def wrap(config):
        print()
        print_text_in_center_with_border(f" Dataset: {config.data.name} ")
        print()
        func(config)
        print()
        print_text_in_center_with_border(" Done! ")

    return wrap
