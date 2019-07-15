import torch
import torch.nn as nn

from torchvision.models import resnet34
from pretrainedmodels import se_resnext50_32x4d


def remove_batchnorm2d(module):
    # TODO: save pretrained parameters
    for child_name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, child_name, nn.Identity())
        else:
            remove_batchnorm2d(child)

    return module


class Encoder(nn.Module):
    def __init__(self, embedding_size, kernel_size, stride=None, backbone_type='resnet34'):
        super().__init__()

        if backbone_type not in ['resnet34', 'seresnext50']:
            raise ValueError(f'Wrong backbone type: expected one of ["resnet34", "seresnext50"], got {backbone_type}')

        # TODO: pretrained | with batchnorm
        if backbone_type == 'resnet34':
            self.backbone = resnet34(pretrained=False)
        elif backbone_type == 'seresnext50':
            self.backbone = se_resnext50_32x4d(pretrained=None)

        if hasattr(self.backbone, "avgpool"):
            self.backbone.avgpool = nn.AdaptiveAvgPool2d(1)
        elif hasattr(self.backbone, "avg_pool"):
            self.backbone.avg_pool = nn.AdaptiveAvgPool2d(1)

        if hasattr(self.backbone, "maxpool"):
            self.backbone.maxpool = nn.Identity()
        elif hasattr(self.backbone, "layer0"):
            self.backbone.layer0[-1] = nn.Identity()

        if hasattr(self.backbone, "fc"):
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, embedding_size)
        elif hasattr(self.backbone, "last_linear"):
            self.backbone.last_linear = nn.Linear(self.backbone.last_linear.in_features, embedding_size)

        self.backbone = remove_batchnorm2d(self.backbone)

        self.kernel_size = kernel_size
        self.stride = kernel_size // 2 if stride is None else stride

    def forward(self, x, average=False):
        x = x.permute((0, 2, 3, 1))
        x = x.unfold(1, self.kernel_size, self.stride).unfold(2, self.kernel_size, self.stride)
        crops = x.contiguous()

        b, r, c, f, h, w = crops.shape
        crops = crops.view(-1, f, h, w)
        embeddings = self.backbone(crops)
        embeddings = embeddings.view(b, r, c, -1)

        if average:
            embeddings = embeddings.mean(dim=(1, 2))
        else:
            embeddings = embeddings.permute((0, 3, 1, 2))

        return embeddings


class LinearPredictor(nn.Module):
    def __init__(self, in_channels, out_channels, n_predictions=1):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv2d(in_channels, out_channels, 1, bias=False) for _ in range(n_predictions)])

    def forward(self, contexts):
        predictions = [conv(contexts) for conv in self.convs]
        return predictions

class ReccurentPredictor(nn.Module):
    def __init__(self, in_channels, out_channels, n_predictions=1):
        super().__init__()
        self.n_predictions = n_predictions
        self.rnn = nn.GRU(in_channels, out_channels, num_layers=1, batch_first=True)

    def forward(self, contexts):
        b, _, h, w = contexts.shape
        contexts = contexts.permute((0, 2, 3, 1)).contiguous().view(b * h * w, 1, -1)
        hiden_state = torch.zeros(1, b * h * w, self.rnn.hidden_size, dtype=torch.float32, device=contexts.device)

        predictions = []
        for _ in range(self.n_predictions):
            outputs, hiden_state = self.rnn(contexts, hiden_state)
            outputs = outputs.view(b, h, w, -1).permute((0, 3, 1, 2)).contiguous()
            predictions.append(outputs)

        return predictions
