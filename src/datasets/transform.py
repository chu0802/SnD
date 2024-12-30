import torch
from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from open_clip.transform import PreprocessCfg, image_transform_v2
from torchvision.transforms import (
    CenterCrop,
    Compose,
    ConvertImageDtype,
    InterpolationMode,
    Normalize,
    PILToTensor,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

DEFAULT_PREPROCESS_CONFIG = {
    "size": (224, 224),
    "mode": "RGB",
    "mean": OPENAI_DATASET_MEAN,
    "std": OPENAI_DATASET_STD,
    "interpolation": "bicubic",
    "resize_mode": "shortest",
    "fill_color": 0,
}


RAW_TRANSFORM = Compose(
    [
        CenterCrop(224),
        PILToTensor(),
        ConvertImageDtype(torch.float),
    ]
)


def _convert_to_rgb(image):
    return image.convert("RGB")


def original_clip_transform(n_px: int = 224, is_train: bool = False):
    normalize = Normalize(
        (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    )
    if is_train:
        return Compose(
            [
                RandomResizedCrop(
                    n_px, scale=(0.9, 1.0), interpolation=InterpolationMode.BICUBIC
                ),
                _convert_to_rgb,
                ToTensor(),
                normalize,
            ]
        )
    else:
        return Compose(
            [
                Resize(n_px, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(n_px),
                _convert_to_rgb,
                ToTensor(),
                normalize,
            ]
        )


def load_transform(config, model_preprocess_config=None):
    if model_preprocess_config is None:
        model_preprocess_config = DEFAULT_PREPROCESS_CONFIG

    if config.data.get("use_original_clip_transform", False):
        train_transform = original_clip_transform(is_train=True)
        eval_transform = original_clip_transform(is_train=False)
    else:
        pp_cfg = PreprocessCfg(**model_preprocess_config)

        train_transform = image_transform_v2(
            pp_cfg,
            is_train=True,
        )

        eval_transform = image_transform_v2(
            pp_cfg,
            is_train=False,
        )

    return train_transform, eval_transform
