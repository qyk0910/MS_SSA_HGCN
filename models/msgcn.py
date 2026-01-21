# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import mne


def compute_electrode_positions(num_nodes=18):
    channel_names = ['AF7', 'FT7', 'TP7', 'PO7', 'AF3', 'FC3',
                    'CP3', 'PO3', 'FCz', 'CPz', 'AF4', 'FC4',
                    'CP4', 'PO4', 'AF8', 'FT8', 'TP8', 'PO8']

    montage = mne.channels.make_standard_montage('standard_1020')
    ch_list = montage.ch_names
    dig_list = montage.dig

    position_matrix = []
    for ch_name in channel_names:
        idx = ch_list.index(ch_name)
        position_matrix.append(dig_list[idx]['r'])

    position_matrix = np.array(position_matrix)

    adjacency = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                dist = np.sqrt(np.sum((position_matrix[i] - position_matrix[j]) ** 2))
                if dist > 0:
                    adjacency[i][j] = 1.0 / dist

    return torch.from_numpy(adjacency).float()


class GraphConstructor(nn.Module):
    def __init__(self, num_nodes, hidden_dim, device='cuda'):
        super(GraphConstructor, self).__init__()
        self.num_nodes = num_nodes
        self.device = device

        self.adj_weights = nn.Parameter(torch.eye(num_nodes, dtype=torch.float32))

        self.similarity_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ELU(inplace=False),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.pos_factor = nn.Parameter(torch.tensor(0.5))

        if num_nodes == 18:
            self.pos_encoding = compute_electrode_positions(num_nodes).to(device)
        else:
            self.pos_encoding = nn.Parameter(torch.eye(num_nodes, dtype=torch.float32))

    def compute_similarity(self, x):
        batch_size, dim, num_nodes = x.shape
        x = x.transpose(1, 2)

        node_i = x.unsqueeze(2).expand(-1, -1, num_nodes, -1)
        node_j = x.unsqueeze(1).expand(-1, num_nodes, -1, -1)

        node_pairs = torch.cat([node_i, node_j], dim=-1)
        similarity = self.similarity_net(
            node_pairs.view(-1, dim * 2)
        ).view(batch_size, num_nodes, num_nodes)

        return similarity

    def forward(self, x):
        static_adj = F.softmax(self.adj_weights, dim=1)

        dynamic_adj = self.compute_similarity(x)

        pos_adj = F.softmax(self.pos_encoding, dim=1)

        combined_adj = static_adj + dynamic_adj + self.pos_factor * pos_adj
        combined_adj = F.softmax(combined_adj, dim=2)

        degree = torch.sum(combined_adj, dim=2)
        degree_inv_sqrt = torch.diag_embed(torch.pow(degree + 1e-8, -0.5))
        normalized_adj = torch.bmm(
            torch.bmm(degree_inv_sqrt, combined_adj), degree_inv_sqrt
        )

        return normalized_adj


class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GraphConvolutionLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adjacency):
        x = x.transpose(1, 2)
        support = torch.bmm(adjacency, x)
        output = torch.matmul(support, self.weight) + self.bias
        return output.transpose(1, 2)


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, dim, num_heads=4):
        super(MultiHeadAttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        return x


class MSGCN(nn.Module):
    def __init__(self, dim, num_channels, reduction=16, device='cuda'):
        super(MSGCN, self).__init__()
        self.device = device
        self.reduction = reduction
        self.num_channels = num_channels
        self.dim = dim

        self.global_graph = GraphConstructor(5, dim, device)
        self.global_gcn = GraphConvolutionLayer(dim, dim)

        self.local_graphs = nn.ModuleList([
            GraphConstructor(num_channels, dim, device) for _ in range(5)
        ])
        self.local_gcns = nn.ModuleList([
            GraphConvolutionLayer(dim, dim) for _ in range(5)
        ])

        self.cross_band_attn = MultiHeadAttentionLayer(dim, num_heads=4)

        self.importance_net = nn.ModuleList([
            nn.Linear(num_channels, 1) for _ in range(5)
        ])

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
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, feature):
        batch_size = feature.size(0)

        band_features = torch.chunk(feature, 5, dim=1)

        local_processed = []
        band_representations = []

        for i, (feat, graph, gcn) in enumerate(
            zip(band_features, self.local_graphs, self.local_gcns)
        ):
            band_feat = feat.squeeze(2)
            local_adj = graph(band_feat)
            local_out = gcn(band_feat, local_adj)
            local_processed.append(local_out)

            representative = torch.mean(local_out, dim=-1)
            band_representations.append(representative)

        scale_nodes = torch.stack(band_representations, dim=1)

        scale_enhanced = self.cross_band_attn(scale_nodes)

        global_adj = self.global_graph(scale_enhanced.transpose(1, 2))
        scale_gcn_out = self.global_gcn(scale_enhanced.transpose(1, 2), global_adj).transpose(1, 2)

        enhanced_locals = []
        for i, local_feat in enumerate(local_processed):
            scale_info = scale_gcn_out[:, i, :].unsqueeze(-1)
            enhanced = local_feat * scale_info
            enhanced_locals.append(enhanced.unsqueeze(2))

        enhanced_feature = torch.cat(enhanced_locals, dim=1)

        A, B, C, D, E = enhanced_feature.split(self.dim, 1)
        stacked = torch.cat((A, B, C, D, E), dim=2)

        bands = [stacked[:, :, i, :] for i in range(5)]

        output = torch.cat([
            self.importance_net[i](band.view(batch_size, self.dim, -1)).unsqueeze(-1)
            for i, band in enumerate(bands)
        ], dim=1)

        return output, global_adj, torch.stack(local_processed, dim=1)
