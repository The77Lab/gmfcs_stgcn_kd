import torch

import torch.nn.functional as F
import torch.nn as nn

import os
import sys

# Get the path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the path of the parent directory
parent_dir = os.path.dirname(script_dir)
# Add the parent directory to the Python path
sys.path.append(parent_dir)

from models.stgcn import STGCNBlock
from utils.graph import Graph



def normal_init(module: nn.Module,
                mean: float = 0,
                std: float = 1,
                bias: float = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class Small_STGCN(nn.Module):
    def __init__(self,
                 graph_cfg,
                 in_channels=3,
                 base_channels=64,
                 block_config = 0,
                 linear_config = 0):
        super(Small_STGCN, self).__init__()
        self.graph = Graph(**graph_cfg)

        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))

        self.block_config = block_config
        self.linear_config = linear_config
        if block_config == 0:
            self.stgcn_block0 = STGCNBlock(in_channels, base_channels, A.clone(), 1, residual=False)

            self.stgcn_block1 = STGCNBlock(base_channels, base_channels, A.clone(), 1, residual=True)
            self.stgcn_block2 = STGCNBlock(base_channels, base_channels*2, A.clone(), stride=2, residual=True)
            self.stgcn_block3 = STGCNBlock(base_channels*2, base_channels*4, A.clone(), stride=2, residual=True)
        elif block_config == 1:
            self.stgcn_block1 = STGCNBlock(in_channels, base_channels, A.clone(), 1, residual=False)
            self.stgcn_block2 = STGCNBlock(base_channels, base_channels*2, A.clone(), stride=2, residual=True)
            self.stgcn_block3 = STGCNBlock(base_channels*2, base_channels*4, A.clone(), stride=2, residual=True)
        elif block_config == 2:
            self.stgcn_block2 = STGCNBlock(in_channels, base_channels*2, A.clone(), stride=2, residual=False)
            self.stgcn_block3 = STGCNBlock(base_channels*2, base_channels*4, A.clone(), stride=2, residual=True)
        elif block_config == 3:
            self.stgcn_block3 = STGCNBlock(in_channels, base_channels*4, A.clone(), stride=4, residual=False)

        self.pool = nn.MaxPool2d((31, 17))

        if linear_config == 0:
            self.linear1 = nn.Linear(base_channels*4, 512)
            self.fc_cls = nn.Linear(512, 4)
        else:
            self.fc_cls = nn.Linear(base_channels*4, 4)

        self.dropout = nn.Dropout(p=0.5)
        if linear_config == 0:
            normal_init(self.linear1, std=0.01)
        normal_init(self.fc_cls, std=0.01)


    def forward(self, x):
        N, M, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = self.data_bn(x.view(N*M, V*C, T))
        x = x.view(N, M, V, C, T).permute(0,1,3, 4,2).contiguous().view(N*M, C, T, V)
        if self.block_config == 0:
            stage_0_out = self.stgcn_block0(x)
            stage_1_out = self.stgcn_block1(stage_0_out)
            stage_2_out = self.stgcn_block2(stage_1_out)
            stage_3_out = self.stgcn_block3(stage_2_out)
        elif self.block_config == 1:
            stage_1_out = self.stgcn_block1(x)
            stage_2_out = self.stgcn_block2(stage_1_out)
            stage_3_out = self.stgcn_block3(stage_2_out)
        elif self.block_config == 2:
            stage_2_out = self.stgcn_block2(x)
            stage_3_out = self.stgcn_block3(stage_2_out)
        elif self.block_config == 3:
            stage_3_out = self.stgcn_block3(x)
            
        x = stage_3_out.reshape((N, M) + stage_3_out.shape[1:])
        
        N, M, C, T, V = x.shape
        # print(x.shape)
        x = x.reshape(N, C, T, V)
        x = self.pool(x)
        x = x.reshape(N, C)

        if self.linear_config == 0:
            x = self.linear1(x)
            x = self.dropout(x)
            x = F.relu(x)
            x = self.fc_cls(x)
        else:
            x = self.fc_cls(x)
        if self.block_config == 0:
            return stage_0_out, stage_1_out, stage_2_out, stage_3_out, x
        elif self.block_config == 1:
            return stage_1_out, stage_2_out, stage_3_out, x
        elif self.block_config == 2:
            return stage_2_out, stage_3_out, x
        elif self.block_config == 3:
            return stage_3_out, x
        # return stage_1_out, stage_2_out, stage_3_out, x
    

class Small_STGCN_Parkinson(nn.Module):
    def __init__(self,
                 graph_cfg,
                 in_channels=3,
                 base_channels=64,
                 block_config = 0,
                 linear_config = 0):
        super(Small_STGCN_Parkinson, self).__init__()
        self.graph = Graph(**graph_cfg)

        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))

        self.block_config = block_config
        self.linear_config = linear_config
        if block_config == 0:
            self.stgcn_block0 = STGCNBlock(in_channels, base_channels, A.clone(), 1, residual=False)

            self.stgcn_block1 = STGCNBlock(base_channels, base_channels, A.clone(), 1, residual=True)
        elif block_config == 1:
            self.stgcn_block1 = STGCNBlock(in_channels, base_channels, A.clone(), 1, residual=False)
        
        self.stgcn_block2 = STGCNBlock(base_channels, base_channels*2, A.clone(), stride=2, residual=True)
        self.stgcn_block3 = STGCNBlock(base_channels*2, base_channels*4, A.clone(), stride=2, residual=True)

        self.pool = nn.MaxPool2d((15, 17))

        if linear_config == 0:
            self.linear1 = nn.Linear(base_channels*4, 512)
            self.udprs_head = nn.Linear(512, 1)
            self.best_head = nn.Linear(512, 1)
            
        else:
            self.udprs_head = nn.Linear(base_channels*4, 1)
            self.best_head = nn.Linear(base_channels*4, 1)

        self.dropout = nn.Dropout(p=0.5)
        if linear_config == 0:
            normal_init(self.linear1, std=0.01)
        normal_init(self.udprs_head, std=0.01)
        normal_init(self.best_head, std=0.01)


    def forward(self, x):
        N, M, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = self.data_bn(x.view(N*M, V*C, T))
        x = x.view(N, M, V, C, T).permute(0,1,3, 4,2).contiguous().view(N*M, C, T, V)
        if self.block_config == 0:
            stage_0_out = self.stgcn_block0(x)
            stage_1_out = self.stgcn_block1(stage_0_out)
        elif self.block_config == 1:
            stage_1_out = self.stgcn_block1(x)
        stage_2_out = self.stgcn_block2(stage_1_out)
        stage_3_out = self.stgcn_block3(stage_2_out)
        x = stage_3_out.reshape((N, M) + stage_3_out.shape[1:])
        
        N, M, C, T, V = x.shape
        # print(x.shape)
        x = x.reshape(N, C, T, V)
        x = self.pool(x)
        x = x.reshape(N, C)

        if self.linear_config == 0:
            x = self.linear1(x)
            x = self.dropout(x)
            x = F.relu(x)
            udprs_score = self.udprs_head(x)
            best_score = self.best_head(x)
        else:
            udprs_score = self.udprs_head(x)
            best_score = self.best_head(x)
        if self.block_config == 0:
            return stage_0_out, stage_1_out, stage_2_out, stage_3_out, udprs_score, best_score
        elif self.block_config == 1:
            return stage_1_out, stage_2_out, stage_3_out, udprs_score, best_score
        # return stage_1_out, stage_2_out, stage_3_out, x