import torch
import torch.nn as nn


class WeightedBCELoss(nn.Module):

    def __init__(self, pos_weight):

        super().__init__()

        self.loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight
        )

    def forward(self, logits, labels):

        loss = self.loss_fn(logits, labels)

        return loss
