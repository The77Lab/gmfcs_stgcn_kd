from stgcn import STGCNBlock
from utils.graph import Graph
import torch
import torch.nn as nn
import torch.nn.functional as F

def normal_init(module: nn.Module,
                mean: float = 0,
                std: float = 1,
                bias: float = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class STGCN_2(nn.Module):
    def __init__(self, num_classes=4, latent_dim=512, in_channels=256, dropout=0.4, init_std=0.01):
        super().__init__()
        graph_config = {
            'layout': 'coco',
            'strategy': 'spatial'
        }
        self.graph = Graph(**graph_config)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.data_bn = nn.BatchNorm1d(3 * A.size(1))
        self.stgcn_block_1 = STGCNBlock(3, 64, A, 1, residual=False)
        self.stgcn_block_2 = STGCNBlock(64, 64, A, 1)

        self.pool = nn.MaxPool2d((31, 17))

        # latent dim 512
        self.linear1 = nn.Linear(512, 64)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(64, 4)

        
    def forward(self, x):
        # x shape batch, person, timestamp, vertex, channel
        N, M, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        # x shape batch, person, vertex, channel, timestamp
        # N, M, V, C, T
        x = x.reshape(N * M, V * C, T)
        x = self.data_bn(x)

        x = x.reshape(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().reshape(N * M, C, T, V)
        # N, M, C, T, V
        x = self.stgcn_block_1(x)
        x = self.stgcn_block_2(x)
        x = x.reshape((N, M) + x.shape[1:])
        
        x = self.pool(x)
        
        return x