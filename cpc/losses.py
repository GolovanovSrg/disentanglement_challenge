import torch
import torch.nn as nn


class CPCLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, predictions, targets):
        predictions = predictions.view(*predictions.shape[:2], -1)
        targets = targets.view(*targets.shape[:2], -1)

        scores = torch.bmm(targets.permute((0, 2, 1)), predictions)
        labels = torch.arange(scores.shape[-1], dtype=torch.long, device=scores.device)
        labels = labels.unsqueeze(0).expand(scores.shape[0], -1)

        loss = self.criterion(scores, labels)

        return loss
