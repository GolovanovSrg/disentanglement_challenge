import torch
import torch.nn as nn


class CPCLoss(nn.Module):
    def __init__(self, n_negative=25, reduction='mean'):
        super().__init__()
        self.n_negative = n_negative
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, predictions, targets):
        predictions = predictions.view(*predictions.shape[:2], -1)
        targets = targets.view(*targets.shape[:2], -1)

        scores = [(predictions * targets).sum(dim=1)]
        for _ in range(self.n_negative):
            batch_perm_idxs = torch.randperm(predictions.shape[0])
            patch_perm_idxs = torch.randperm(predictions.shape[-1])
            scores.append((predictions * targets[batch_perm_idxs, ...][..., patch_perm_idxs]).sum(dim=1))

        scores = torch.stack(scores, dim=1)
        labels = torch.zeros(scores.shape[0], scores.shape[-1], dtype=torch.long, device=scores.device)

        loss = self.criterion(scores, labels)

        return loss
