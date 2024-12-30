import torch
import torch.nn as nn

from src.trainer.base_trainer import BaseKDTrainer
from src.trainer.utils import L2Loss


class SnDTrainer(BaseKDTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prev_teacher_model.eval()
        self.feature_criterion = L2Loss(reduce=None, square=False)
        self.num_valid_prev_data = 0

    @property
    def prev_teacher_model(self):
        return self._teachers["prev"]

    def snd_loss(
        self,
        images,
        labels,
        ratio_prev=9,
        ratio_pretrained=0.5,
        threshold=0.2,
        scale=6,
        label_smoothing=0.0,
    ):
        ref_images, _ = self.get_ref_data(self.ref_loader)
        base_loss, loss_dict = self.base_loss(
            images, labels, label_smoothing=label_smoothing
        )

        student_ref_image_embedding = self.unwrapped_model(self.train_model).encode(
            images=ref_images
        )

        with torch.no_grad():
            (
                pretrained_teacher_ref_image_embedding,
                _,
                _,
            ) = self.pretrained_teacher_model(ref_images, get_features=True)

            (
                prev_teacher_ref_image_embedding,
                _,
                _,
            ) = self.prev_teacher_model(ref_images, get_features=True)

        pre_scores = torch.norm(
            pretrained_teacher_ref_image_embedding - prev_teacher_ref_image_embedding,
            dim=-1,
        )

        self.num_valid_prev_data += (pre_scores > threshold).float().sum().item()

        scaled_scores = scale * (pre_scores - threshold)

        scores = nn.functional.sigmoid(scaled_scores).reshape(-1, 1)

        pretrained_kd_loss = self._get_kd_loss(
            student_ref_image_embedding,
            pretrained_teacher_ref_image_embedding,
            feature_criterion=self.feature_criterion,
        )
        prev_kd_loss = self._get_kd_loss(
            student_ref_image_embedding,
            prev_teacher_ref_image_embedding,
            feature_criterion=self.feature_criterion,
        )

        prev_kd_loss = (scores * prev_kd_loss).mean()

        pretrained_kd_loss = ((1 - scores) * pretrained_kd_loss).mean()

        return (
            base_loss
            + ratio_prev * prev_kd_loss
            + ratio_pretrained * pretrained_kd_loss,
            {
                **loss_dict,
                "prev_kd_loss": prev_kd_loss.item(),
                "pretrained_kd_loss": pretrained_kd_loss.item(),
                "num_valid_prev_data": self.num_valid_prev_data,
            },
        )
