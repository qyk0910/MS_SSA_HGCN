# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from .mtfe import MTFE
from .mlafs import FeatureSelectionModule
from .msgcn import MSGCN


class ClassificationHead(nn.Module):
    def __init__(self):
        super(ClassificationHead, self).__init__()
        self.conv1 = nn.Conv2d(
            96 * 5, 48, kernel_size=(1, 1),
            stride=1, padding=(0, 0), bias=True
        )
        self.bn1 = nn.BatchNorm2d(48)
        self.conv2 = nn.Conv2d(
            48, 2, kernel_size=(1, 1),
            stride=1, padding=(0, 0), bias=True
        )
        self.elu = nn.ELU(inplace=False)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        output = self.conv2(x)
        return output


class MS_SSA_HGCN(nn.Module):
    def __init__(self, feature_dim=96, input_channels=18, hidden_dim=6,
                 device_id=1, use_position=False):
        super(MS_SSA_HGCN, self).__init__()
        self.feature_dim = feature_dim
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.device_id = device_id
        self.use_position = use_position

        self.mtfe = MTFE(out_channels=4, num_layers=1)

        base_channels = 4 * (2 ** 1) + 1
        self.mlafs = FeatureSelectionModule(
            input_dim=base_channels * 3,
            num_heads=9,
            output_dim=feature_dim
        )

        self.msgcn = MSGCN(
            dim=feature_dim,
            num_channels=input_channels,
            reduction=16,
            device=f'cuda:{device_id}'
        )

        self.classifier = ClassificationHead()

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x):
        gamma, beta, alpha, theta, delta = self.mtfe(x)

        combined_features = [gamma, beta, alpha, theta, delta]
        mtfe_output = torch.cat(combined_features, dim=1)

        mlafs_output = self.mlafs(combined_features)

        batch_size = mlafs_output.size(0)
        mlafs_output = mlafs_output.view(batch_size, self.feature_dim, 5, self.input_channels)
        mlafs_output = mlafs_output.permute(0, 1, 3, 2).contiguous()
        mlafs_output = mlafs_output.view(batch_size, self.feature_dim * 5, 1, self.input_channels)

        graph_output, global_adj, local_adjs = self.msgcn(mlafs_output)

        predictions = self.classifier(graph_output).squeeze()

        return predictions, [
            mtfe_output,
            graph_output,
            global_adj,
            local_adjs
        ]
