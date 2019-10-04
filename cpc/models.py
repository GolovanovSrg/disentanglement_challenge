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
        self.q_proj = nn.Linear(n_features, n_features, bias=False)
        self.k_proj = nn.Linear(n_features // n_heads, n_features, bias=False)
        self.v_proj = nn.Linear(n_features // n_heads, n_features, bias=False)
        self.embedding = nn.Parameter(torch.normal(mean=torch.zeros(n_embedding, n_features // n_heads), std=0.5))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        assert x.dim() == 2

        q = self.q_proj(x).unsqueeze(1)
        k = self.k_proj(self.embedding).unsqueeze(0).expand(x.shape[0], -1, -1)
        v = self.v_proj(self.embedding).unsqueeze(0).expand(x.shape[0], -1, -1)

        q = torch.stack(torch.split(q, self.n_features // self.n_heads, dim=2), dim=0)
        k = torch.stack(torch.split(k, self.n_features // self.n_heads, dim=2), dim=0)
        v = torch.stack(torch.split(v, self.n_features // self.n_heads, dim=2), dim=0)

        w = torch.matmul(q, k.transpose(2, 3)) 
        w = w / ((self.n_features // self.n_heads) ** 0.5)
        w = F.softmax(w, dim=3)
        w = self.dropout(w)

        out = torch.matmul(w, v)
        out = torch.cat(torch.split(out, 1, dim=0), dim=3)
        out = out.squeeze(0).squeeze(1)

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
    def __init__(self, embedding_size, kernel_size, stride=None, pretrained=False, backbone_type='resnet34'):
        super().__init__()

        if backbone_type not in ['resnet18', 'resnet34', 'seresnext50']:
            raise ValueError(f'Wrong backbone type: expected one of ["resnet34", "seresnext50"], got {backbone_type}')

        if backbone_type == 'resnet18':
            self.backbone = resnet18(pretrained=pretrained)
        elif backbone_type == 'resnet34':
            self.backbone = resnet34(pretrained=pretrained)
        elif backbone_type == 'seresnext50':
            self.backbone = se_resnext50_32x4d(pretrained=('imagenet' if pretrained else None))

        self.backbone = remove_batchnorm2d(self.backbone)

        mean = [0.485, 0.456, 0.406] if pretrained else [0.5, 0.5, 0.5]
        std = [0.229, 0.224, 0.225] if pretrained else [0.5, 0.5, 0.5]
        self.register_buffer('mean', torch.tensor(mean))
        self.register_buffer('std', torch.tensor(std))

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
            self.backbone.last_linear = nn.Sequential(nn.Linear(self.backbone.last_linear.in_features, embedding_size))

        self.embedding_size = embedding_size
        self.kernel_size = kernel_size
        self.stride = kernel_size // 2 if stride is None else stride

    def forward(self, x, average=False):
        x = x.permute((0, 2, 3, 1))
        x = (x - self.mean) / self.std
        x = x.unfold(1, self.kernel_size, self.stride).unfold(2, self.kernel_size, self.stride)
        crops = x.contiguous()
        b, r, c, f, h, w = crops.shape
        crops = crops.view(-1, f, h, w)
        
        embeddings = self.backbone(crops)
        embeddings = embeddings.view(b, r, c, -1)
        embeddings = embeddings.permute((0, 3, 1, 2))

        if average:
            embeddings = embeddings.mean(dim=(2, 3))

        return embeddings


class LinearPredictor(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU(inplace=True), n_predictions=1):
        super().__init__()
        self.convs = nn.ModuleList([nn.Sequential(nn.Conv2d(in_channels, out_channels, 1),
                                                  activation,
                                                  nn.Conv2d(out_channels, out_channels, 1))
                                    for _ in range(n_predictions)])

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


class RepresentationExtractor(nn.Module):
    def __init__(self, encoder, autoregressor, batch_size=64):
        super().__init__()
        self.encoder = encoder
        self.autoregressor = autoregressor
        self.batch_size = batch_size

    def forward(self, x):
        result = []
        chunker = (x[pos:pos + self.batch_size] for pos in range(0, len(x), self.batch_size))
        for chunk in chunker:
            chunk = self.encoder(chunk, average=True)
            #chunk = self.autoregressor(chunk)
            #chunk = chunk[:, :, -1, -1]
            result.append(chunk)
            
        return torch.cat(result, dim=0)
