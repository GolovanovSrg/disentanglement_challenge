import torch
import torch.nn as nn

from pretrainedmodels import resnet34, se_resnext50_32x4d


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
            self.backbone = resnet34(pretrained=None)
        elif backbone_type == 'seresnext50':
            self.backbone = se_resnext50_32x4d(pretrained=None)

        if hasattr(self.backbone, "avgpool"):
            self.backbone.avgpool = nn.AdaptiveAvgPool2d(1)
        elif hasattr(self.backbone, "avg_pool"):
            self.backbone.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.backbone.last_linear = nn.Linear(self.backbone.last_linear.in_features, embedding_size)
        self.backbone = remove_batchnorm2d(self.backbone)

        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride

    def get_crops(self, x):
        x = x.unfold(1, self.kernel_size, self.stride)
        x = x.unfold(2, self.kernel_size, self.stride)
        return x.contiguous()

    def forward(self, x, average=False):
        x = x.permute((0, 2, 3, 1))
        crops = Encoder.get_crops(x)
        b, r, c, f, h, w = crops.shape
        crops = crops.view(-1, f, h, w)
        embeddings = self.backbone(crops)
        embeddings = embeddings.view(b, r, c, -1)

        if average:
            embeddings = embeddings.mean(dim=(1, 2))
        else:
            embeddings = embeddings.permute((0, 3, 1, 2))

        return embeddings


class Predictor(nn.Module):
    def __init__(self, in_channels, out_channels, n_predictions=1):
        super().__init__()

        self.convs = nn.ModuleList()
        for _ in range(n_predictions):
            conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            self.convs.append(conv)

    def forward(self, contexts):
        return [conv(contexts) for conv in self.convs]
