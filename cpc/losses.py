import torch
import torch.nn as nn


class CPCLoss(nn.Module):
    def __init__(self):
        super.__init__()

        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, predictions, targets):
        predictions = predictions.view(*predictions.shape[:2], -1)  # b, z, n
        targets = targets.view(*targets.shape[:2], -1)  # b, z, n

        scores = torch.bmm(targets.pemute((0, 2, 1)), predictions)  # b, n, n
        labels = torch.arange(scores.shape[-1], dtype=torch.long, device=scores.device)
        labels = labels.unsqueeze(0).expand(scores.shape[0], -1)

        loss = self.criterion(scores, labels)

        return loss
