import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.utils import disabled_train


SIMPLE_TEMPLATE = lambda c: f"a photo of a {c}."

# In PureClip model, the text-encoder is involved in the training progress.
# FIXME: Directly remove logit scale before loading previous model might result in an error.
class PureClip(nn.Module):
    def __init__(
        self,
        model_name,
        class_name_list,
        freeze_classification_head=False,
        device="cuda",
    ):
        super().__init__()
        self.model = open_clip.create_model_from_pretrained(
            model_name,
            pretrained="openai",
            return_transform=False,
        ).to(device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.template = SIMPLE_TEMPLATE
        self.device = device
        self.class_tokens = self.tokenize(class_name_list)

        self.freeze_classification_head = freeze_classification_head

        if self.freeze_classification_head:

            for name, p in self.model.named_parameters():
                if "visual" not in name:
                    p.requires_grad = False
            self.model.transformer.eval()
            self.model.transformer.train = disabled_train

    @property
    def preprocess_config(self):
        return self.model.visual.preprocess_cfg

    def tokenize(self, texts, device="cuda"):
        return self.tokenizer([self.template(t) for t in texts]).to(device)

    @torch.no_grad()
    def get_class_embedding(self, class_name_list, device="cuda"):
        tokens = self.tokenizer([self.template(t) for t in class_name_list]).to(device)
        text_embedding = self.model.encode_text(tokens)
        return F.normalize(text_embedding)

    def encode(self, images=None, text=None, normalize=True):
        if images is None:
            text_embeddings = self.model.encode_text(text)
            return F.normalize(text_embeddings) if normalize else text_embeddings
        if text is None:
            image_embeddings = self.model.encode_image(images)
            return F.normalize(image_embeddings) if normalize else image_embeddings

    # to fit the format of clip-classifier, we send a list of data to pure-clip if text is neeeded.
    def forward(self, images, text=None, normalize=True, get_features=False):
        if text is None:
            text = self.class_tokens

        image_embeddings = self.encode(images=images, normalize=normalize)
        text_embeddings = self.encode(text=text, normalize=normalize)

        if get_features:
            return image_embeddings, text_embeddings, self.model.logit_scale.exp()

        res = image_embeddings @ text_embeddings.t()

        res *= self.model.logit_scale.exp()

        return res

    def get_params(self):
        exclude_param = "logit_scale"
        return [
            {
                "params": [
                    p
                    for k, p in self.model.named_parameters()
                    if p.requires_grad and exclude_param not in k
                ]
            }
        ]

    def get_state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)


def get_model(
    config,
    class_name_list,
    pretrained=False,
    freeze=False,
    device="cuda",
):

    model_config = config.model
    
    model = PureClip(
        model_config.vit_base,
        class_name_list,
        freeze_classification_head=model_config.get(
            "freeze_classification_head", False
        ),
        device=device,
    )

    # then load from a checkpoint if not pre-trained
    if model_config.pretrained != "openai" and not pretrained:
        model.load_state_dict(torch.load(model_config.pretrained)["model"])

    model = model.to(device)

    if freeze:
        for _, v in model.named_parameters():
            v.requires_grad = False
        model.eval()

    if config.task.get("distributed", False) and not freeze:
        model = nn.parallel.DistributedDataParallel(model)

    return model
