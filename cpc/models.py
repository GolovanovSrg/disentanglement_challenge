import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, resnet34
from pretrainedmodels import se_resnext50_32x4d


class MultiheadEmbedding(nn.Module):
    def __init__(self, n_features, n_embedding, n_heads, dropout=0):
        super().__init__()

        assert n_features % n_heads == 0

        self.n_features = n_features
        self.n_heads = n_heads
        self.proj = nn.Linear(n_features, n_features)
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Parameter(torch.empty((n_features, n_embedding)))

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.normal_(self.embedding, std=0.02)

    def forward(self, x):
        assert x.dim() == 2
        x = self.proj(x)
        x = x.view(-1, self.n_heads, self.n_features // self.n_heads).unsqueeze(2)  # b, h, 1, f
        k = self.embedding.view(self.n_heads, self.n_features // self.n_heads, -1).unsqueeze(0)  # 1, h, f, m
        w = torch.matmul(x, k) / math.sqrt(self.n_features // self.n_heads)  # b, h, 1, m
        w = torch.sigmoid(w, dim=-1)
        w = self.dropout(w)
        v = self.embedding.view(self.n_heads, self.n_features // self.n_heads, -1).unsqueeze(0).permute((0, 1, 3, 2))  # 1, h, m, f
        out = torch.matmul(w, v)  # b, h, 1, f
        out = out.view(out.shape[0], -1)

        return out


class BasisEmbedding(nn.Module):
    def __init__(self, n_features, n_embedding):
        super().__init__()
        self.proj = nn.Linear(n_features, n_features)
        self.embedding = nn.Parameter(torch.empty((n_features, n_embedding)))

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.normal_(self.embedding, std=0.02)

    def forward(self, x):
        basis = F.normalize(F.relu(self.embedding), dim=0)
        x = self.proj(x)
        x = x.view(-1, x.shape[-1]).unsqueeze(1)  # b, 1, f
        k = basis.unsqueeze(0)  # 1, f, m
        w = torch.matmul(x, k)  # b, 1, m
        v = basis.unsqueeze(0).permute((0, 2, 1))  # 1, m, f
        out = torch.matmul(w, v)  # b, 1, f
        out = out.squeeze(1)

        return out


def remove_batchnorm2d(module):
    for child_name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            gnorm = nn.GroupNorm(1, child.num_features)
            gnorm.weight.data = child.weight.data
            gnorm.bias.data = child.bias.data
            setattr(module, child_name, gnorm)
        else:
            remove_batchnorm2d(child)

    return module


class Encoder(nn.Module):
    def __init__(self, embedding_size, kernel_size, stride=None, backbone_type='resnet34'):
        super().__init__()

        if backbone_type not in ['resnet18', 'resnet34', 'seresnext50']:
            raise ValueError(f'Wrong backbone type: expected one of ["resnet34", "seresnext50"], got {backbone_type}')

        if backbone_type == 'resnet18':
            self.backbone = resnet18(pretrained=True)
        elif backbone_type == 'resnet34':
            self.backbone = resnet34(pretrained=True)
        elif backbone_type == 'seresnext50':
            self.backbone = se_resnext50_32x4d(pretrained='imagenet')

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

        mapping = []
        for i in range(8):
            mapping.append(nn.ReLU(inplace=True))
            mapping.append(nn.Conv2d(embedding_size, embedding_size, kernel_size=1, groups=1, bias=True))

        self.mapping = nn.Sequential(*mapping)
        self.basis = BasisEmbedding(embedding_size, embedding_size)

    def forward(self, x, average=False):
        x = x.permute((0, 2, 3, 1))
        x = x.unfold(1, self.kernel_size, self.stride).unfold(2, self.kernel_size, self.stride)
        crops = x.contiguous()
        b, r, c, f, h, w = crops.shape
        crops = crops.view(-1, f, h, w)
        
        embeddings = self.backbone(crops)
        embeddings = F.relu(embeddings)
        embeddings = self.basis(embeddings)
        embeddings = embeddings.view(b, r, c, -1)
        embeddings = embeddings.permute((0, 3, 1, 2))

        #embeddings = self.mapping(embeddings)

        if average:
            embeddings = embeddings.mean(dim=(2, 3))

        return embeddings


class LinearPredictor(nn.Module):
    def __init__(self, in_channels, out_channels, n_predictions=1):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv2d(in_channels, out_channels, 1, bias=False) for _ in range(n_predictions)])

    def forward(self, contexts):
        predictions = [F.relu(conv(contexts)) for conv in self.convs]
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
            predictions.append(F.relu(outputs))

        return predictions
