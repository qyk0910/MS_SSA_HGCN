# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from einops import rearrange


class MultiLevelSparseAttention(nn.Module):
    def __init__(self, dim, num_heads=9, output_dim=48, bias=False):
        super(MultiLevelSparseAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.output_dim = output_dim

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1,
            padding=1, groups=dim * 3, bias=bias
        )

        self.project_out = nn.Conv2d(dim, output_dim, kernel_size=1, bias=bias)

        self.attn_weight1 = nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn_weight2 = nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn_weight3 = nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn_weight4 = nn.Parameter(torch.tensor([0.2]), requires_grad=True)

        self.attn_drop = nn.Dropout(0.0)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))

        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, C, _ = q.shape

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        mask1 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        index1 = torch.topk(attn, k=int(C/2), dim=-1, largest=True)[1]
        mask1.scatter_(-1, index1, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))

        mask2 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        index2 = torch.topk(attn, k=int(C*2/3), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index2, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))

        mask3 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        index3 = torch.topk(attn, k=int(C*3/4), dim=-1, largest=True)[1]
        mask3.scatter_(-1, index3, 1.)
        attn3 = torch.where(mask3 > 0, attn, torch.full_like(attn, float('-inf')))

        mask4 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        index4 = torch.topk(attn, k=int(C*4/5), dim=-1, largest=True)[1]
        mask4.scatter_(-1, index4, 1.)
        attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, float('-inf')))

        attn1 = attn1.softmax(dim=-1)
        attn2 = attn2.softmax(dim=-1)
        attn3 = attn3.softmax(dim=-1)
        attn4 = attn4.softmax(dim=-1)

        out1 = attn1 @ v
        out2 = attn2 @ v
        out3 = attn3 @ v
        out4 = attn4 @ v

        out = (out1 * self.attn_weight1 +
               out2 * self.attn_weight2 +
               out3 * self.attn_weight3 +
               out4 * self.attn_weight4)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                       head=self.num_heads, h=h, w=w)

        out = self.project_out(out)

        return out


class FeatureSelectionModule(nn.Module):
    def __init__(self, input_dim, num_heads=9, output_dim=96):
        super(FeatureSelectionModule, self).__init__()
        self.mlafs = MultiLevelSparseAttention(
            dim=input_dim,
            num_heads=num_heads,
            output_dim=output_dim,
            bias=False
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((18, 1))

    def forward(self, features):
        processed_features = []
        for feat in features:
            out = self.mlafs(feat)
            out = self.avg_pool(out)
            processed_features.append(out)

        return torch.cat(processed_features, dim=1)
