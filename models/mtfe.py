# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange


class SpectralAttention(nn.Module):
    def __init__(self, channels):
        super(SpectralAttention, self).__init__()
        self.reduce = nn.Conv2d(channels, channels // 4, kernel_size=1)
        self.expand = nn.Conv2d(channels // 4, channels, kernel_size=1)

    def forward(self, x):
        y = F.adaptive_avg_pool2d(x, 1)
        y = self.reduce(y)
        y = F.relu(y, inplace=True)
        y = self.expand(y)
        y = torch.sigmoid(y)
        return x * y


class WaveletTransform(nn.Module):
    def __init__(self, in_channels):
        super(WaveletTransform, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels * 2, kernel_size=(1, 8),
            stride=(1, 2), padding=0, groups=in_channels, bias=False
        )

        self.spectral_enhance = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels * 4, kernel_size=1),
            nn.BatchNorm2d(in_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 4, in_channels * 2, kernel_size=1),
            nn.BatchNorm2d(in_channels * 2),
            nn.Sigmoid()
        )

        self.freq_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, 1)),
            nn.Conv2d(in_channels * 2, in_channels // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels * 2, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_padded = torch.cat((x[:, :, :, -3:], x, x[:, :, :, 0:3]), 3)

        wavelet_out = self.conv(x_padded)

        enhance = self.spectral_enhance(wavelet_out)
        out = wavelet_out * enhance

        freq_weights = self.freq_attn(out)
        out = out * freq_weights

        low_freq = out[:, 0::2, :, :]
        high_freq = out[:, 1::2, :, :]

        return low_freq, high_freq


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        mid_channels = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).squeeze(-1).squeeze(-1))
        max_out = self.fc(self.max_pool(x).squeeze(-1).squeeze(-1))
        out = self.sigmoid(avg_out + max_out).unsqueeze(-1).unsqueeze(-1)
        return x * out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_pool, max_pool], dim=1)
        attention = self.sigmoid(self.conv(combined))
        return x * attention


class DepthwiseConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super(DepthwiseConv2D, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)


class DepthwiseConv1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseConv1D, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=(1, 3),
            padding=(0, 1), groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x


class StateSpaceLayer(nn.Module):
    def __init__(self, feature_dim, state_dim=32):
        super(StateSpaceLayer, self).__init__()
        self.feature_dim = feature_dim
        self.state_dim = state_dim

        self.A = nn.Parameter(torch.randn(state_dim, state_dim))
        nn.init.normal_(self.A, mean=0.0, std=0.1)

        self.B = nn.Parameter(torch.randn(state_dim))
        nn.init.normal_(self.B, mean=0.0, std=0.1)

        self.C = nn.Parameter(torch.randn(state_dim, feature_dim))
        nn.init.normal_(self.C, mean=0.0, std=0.1)

        self.D = nn.Parameter(torch.randn(feature_dim))
        nn.init.normal_(self.D, mean=0.0, std=0.1)

        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(0.0)

    def forward(self, x):
        batch_size, channels, height, time_steps = x.shape
        x_reshaped = x.reshape(batch_size * height, channels, time_steps)

        state = torch.zeros(batch_size * height, self.state_dim, device=x.device)

        outputs = []
        for t in range(time_steps):
            input_t = x_reshaped[:, :, t]
            state = torch.matmul(state, self.A.T) + self.B.unsqueeze(0)
            output_t = torch.matmul(state, self.C) + input_t * self.D
            output_t = self.activation(output_t)
            outputs.append(output_t)

        output = torch.stack(outputs, dim=-1)
        output = output.reshape(batch_size, channels, height, time_steps)

        output = x + self.dropout(output)

        output_norm = output.permute(0, 2, 3, 1)
        output_norm = self.norm(output_norm.reshape(-1, self.feature_dim))
        output_norm = output_norm.reshape(batch_size, height, time_steps, channels)
        output = output_norm.permute(0, 3, 1, 2)

        return output


class CascadedBilateralSSM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CascadedBilateralSSM, self).__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        self.dw2d_1_1 = DepthwiseConv2D(out_channels, out_channels)
        self.dw2d_1_2 = DepthwiseConv2D(out_channels, out_channels)

        self.channel_attn_1 = ChannelAttention(out_channels)
        self.spatial_attn_1 = SpatialAttention()

        self.dw1d_1 = DepthwiseConv1D(out_channels, out_channels)
        self.ssm_1 = StateSpaceLayer(out_channels)

        self.flip_fn = lambda x: torch.flip(x, dims=[3])

        self.dw2d_2_1 = DepthwiseConv2D(out_channels, out_channels)
        self.dw2d_2_2 = DepthwiseConv2D(out_channels, out_channels)

        self.channel_attn_2 = ChannelAttention(out_channels)
        self.spatial_attn_2 = SpatialAttention()

        self.dw1d_2 = DepthwiseConv1D(out_channels, out_channels)
        self.ssm_2 = StateSpaceLayer(out_channels)

        self.output_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.output_bn = nn.BatchNorm2d(out_channels)
        self.output_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.input_conv(x)

        branch1 = self.dw2d_1_1(x)
        branch2 = self.dw2d_1_2(x)

        channel_branch = branch1 * self.channel_attn_1(branch1)
        channel_branch = self.dw1d_1(channel_branch)
        channel_branch = torch.sigmoid(channel_branch)

        spatial_branch = branch2 * self.spatial_attn_1(branch2)
        spatial_branch = torch.sigmoid(spatial_branch)

        ssm_out = self.ssm_1(channel_branch)
        out1 = ssm_out * spatial_branch

        flipped = self.flip_fn(out1)

        branch1 = self.dw2d_2_1(flipped)
        branch2 = self.dw2d_2_2(flipped)

        channel_branch = branch1 * self.channel_attn_2(branch1)
        channel_branch = self.dw1d_2(channel_branch)
        channel_branch = torch.sigmoid(channel_branch)

        spatial_branch = branch2 * self.spatial_attn_2(branch2)
        spatial_branch = torch.sigmoid(spatial_branch)

        ssm_out = self.ssm_2(channel_branch)
        out2 = ssm_out * spatial_branch

        out2 = self.flip_fn(out2)
        out = self.output_conv(out2)
        out = self.output_bn(out)
        out = self.output_relu(out)

        return out


class TemporalDownsample(nn.Module):
    def __init__(self, channels, kernel_size, stride, padding):
        super(TemporalDownsample, self).__init__()
        self.conv = nn.Conv2d(
            channels, channels, kernel_size=(1, kernel_size),
            stride=(1, stride), padding=(0, padding), bias=False
        )
        self.bn = nn.BatchNorm2d(channels)
        self.elu = nn.ELU(inplace=False)
        self.ssm = CascadedBilateralSSM(channels, channels)

    def forward(self, x):
        x = self.elu(self.bn(self.conv(x)))
        x = self.ssm(x)
        return x


class HaarWaveletDownsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HaarWaveletDownsample, self).__init__()
        try:
            from pytorch_wavelets import DWTForward
            self.dwt = DWTForward(J=1, mode='zero', wave='haar')
        except ImportError:
            self.dwt = None

        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_channels * 4, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, target_h=None, target_w=None):
        if self.dwt is not None:
            yL, yH = self.dwt(x)
            y_HL = yH[0][:, :, 0, ::]
            y_LH = yH[0][:, :, 1, ::]
            y_HH = yH[0][:, :, 2, ::]
            x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        else:
            x = F.avg_pool2d(x, 2)

        x = self.conv_bn_relu(x)

        if target_h is not None and x.size(2) != target_h:
            th = target_h if target_h is not None else x.size(2)
            tw = target_w if target_w is not None else x.size(3)
            x = F.adaptive_avg_pool2d(x, (th, tw))

        return x


class InputProjection(nn.Module):
    def __init__(self, out_channels):
        super(InputProjection, self).__init__()
        self.conv = nn.Conv2d(
            1, out_channels, kernel_size=(1, 3),
            stride=1, padding=(0, 1), groups=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super(ResidualBlock, self).__init__()
        self.expand = nn.Conv2d(
            in_channels, out_channels, kernel_size=1,
            stride=1, padding=0, groups=groups, bias=False
        ) if in_channels != out_channels else None

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=(1, 3),
            stride=1, padding=(0, 1), groups=groups, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=(1, 3),
            stride=1, padding=(0, 1), groups=groups, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = self.expand(x) if self.expand is not None else x
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        return torch.add(x, identity)


class MTFE(nn.Module):
    def __init__(self, out_channels, num_layers=1):
        super(MTFE, self).__init__()
        self.num_layers = num_layers

        layers = [InputProjection(out_channels)]
        for i in range(num_layers):
            layers.append(ResidualBlock(
                int(2 ** i) * out_channels,
                int(2 ** (i + 1)) * out_channels
            ))
        self.embedding = nn.Sequential(*layers)

        base_channels = out_channels * (2 ** num_layers) + 1

        self.msst_gamma = WaveletTransform(base_channels)
        self.msst_beta = WaveletTransform(base_channels)
        self.msst_alpha = WaveletTransform(base_channels)
        self.msst_theta = WaveletTransform(base_channels)
        self.msst_delta = WaveletTransform(base_channels)

        self.mstm_gamma = TemporalDownsample(base_channels, 4, 4, 0)
        self.mstm_beta = TemporalDownsample(base_channels, 8, 8, 0)
        self.mstm_alpha = TemporalDownsample(base_channels, 16, 16, 0)
        self.mstm_theta = TemporalDownsample(base_channels, 32, 32, 0)
        self.mstm_delta = TemporalDownsample(base_channels, 32, 32, 0)

        self.hwd_gamma = HaarWaveletDownsample(base_channels, base_channels)
        self.hwd_beta = HaarWaveletDownsample(base_channels, base_channels)
        self.hwd_alpha = HaarWaveletDownsample(base_channels, base_channels)
        self.hwd_theta = HaarWaveletDownsample(base_channels, base_channels)
        self.hwd_delta = HaarWaveletDownsample(base_channels, base_channels)

    def forward(self, x):
        x_emb = self.embedding(x)
        x_concat = torch.cat((x_emb, x), 1)

        _, gamma = self.msst_gamma(x_concat)
        _, beta = self.msst_beta(x_concat)
        _, alpha = self.msst_alpha(x_concat)
        _, theta = self.msst_theta(x_concat)
        _, delta = self.msst_delta(x_concat)

        target_h = gamma.size(2)
        target_w = gamma.size(3)

        beta = F.adaptive_avg_pool2d(beta, (target_h, target_w))
        alpha = F.adaptive_avg_pool2d(alpha, (target_h, target_w))
        theta = F.adaptive_avg_pool2d(theta, (target_h, target_w))
        delta = F.adaptive_avg_pool2d(delta, (target_h, target_w))

        down_gamma = self.mstm_gamma(x_concat)
        down_gamma = F.adaptive_avg_pool2d(down_gamma, (target_h, target_w))

        down_beta = self.mstm_beta(x_concat)
        down_beta = F.adaptive_avg_pool2d(down_beta, (target_h, target_w))

        down_alpha = self.mstm_alpha(x_concat)
        down_alpha = F.adaptive_avg_pool2d(down_alpha, (target_h, target_w))

        down_theta = self.mstm_theta(x_concat)
        down_theta = F.adaptive_avg_pool2d(down_theta, (target_h, target_w))

        down_delta = self.mstm_delta(x_concat)
        down_delta = F.adaptive_avg_pool2d(down_delta, (target_h, target_w))

        haar_gamma = self.hwd_gamma(x_concat, target_h=target_h, target_w=target_w)
        haar_beta = self.hwd_beta(x_concat, target_h=target_h, target_w=target_w)
        haar_alpha = self.hwd_alpha(x_concat, target_h=target_h, target_w=target_w)
        haar_theta = self.hwd_theta(x_concat, target_h=target_h, target_w=target_w)
        haar_delta = self.hwd_delta(x_concat, target_h=target_h, target_w=target_w)

        gamma = torch.cat([down_gamma, haar_gamma, gamma], 1)
        beta = torch.cat([down_beta, haar_beta, beta], 1)
        alpha = torch.cat([down_alpha, haar_alpha, alpha], 1)
        theta = torch.cat([down_theta, haar_theta, theta], 1)
        delta = torch.cat([down_delta, haar_delta, delta], 1)

        return gamma, beta, alpha, theta, delta
