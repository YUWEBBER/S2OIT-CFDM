"""训练损失：PatchNCE 对比损失与 MMD 分布对齐损失。

两者都作用于 DINOv2 的多层特征图，用于约束输入 SAR 与输出彩色图的结构一致性。
"""

import numpy as np
import torch
import torch.nn as nn


class NCELoss(nn.Module):
    """PatchNCE 对比损失。

    在 DINOv2 的多层特征图上随机采样图像块，把同一空间位置的跨域特征作为正样本、
    同批其它位置作为负样本，迫使生成图像保持输入图像的空间结构。

    Args:
        feature_dim: DINO 特征维度（vits14 为 384）。
        num_layers: 参与计算的特征层数。
        num_patches: 每层采样的图像块数量。
        lambda_nce: 损失权重。
        temperature: InfoNCE 温度系数。
    """

    def __init__(self, feature_dim, num_layers, num_patches=64, lambda_nce=0.2, temperature=0.07, device="cuda"):
        super().__init__()
        self.num_layers = num_layers
        self.num_patches = num_patches
        self.lambda_nce = lambda_nce
        self.temperature = temperature
        # 每个特征层一个两层 MLP，把特征映射到对比空间
        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.ReLU(),
                nn.Linear(feature_dim // 2, feature_dim),
            )
            for _ in range(num_layers)
        ]).to(device)
        for module in self.projectors.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, a=0, mode="fan_in")
                nn.init.constant_(module.bias, 0.0)

    def _project(self, features, patch_ids):
        """在采样位置投影特征并做 L2 归一化。

        Args:
            features: 每层 (B, C, H, W) 的特征图列表。
            patch_ids: (P,) 采样位置索引（在 H*W 内）。

        Returns:
            每层 (B*P, C) 的归一化特征列表。
        """
        projected = []
        for mlp, feature in zip(self.projectors, features):
            tokens = feature.permute(0, 2, 3, 1).flatten(1, 2)      # (B, H*W, C)
            tokens = tokens[:, patch_ids, :].flatten(0, 1)          # (B, P, C) -> (B*P, C)
            tokens = mlp(tokens)
            projected.append(tokens / (tokens.norm(dim=1, keepdim=True) + 1e-7))
        return projected

    def forward(self, dino_encoder, real_image, generated_image):
        """计算对比损失。

        Args:
            dino_encoder: 冻结的 DINOv2 编码器，返回多层特征图。
            real_image: 输入 SAR 图像 (B, 3, H, W)，取值 [-1, 1]。
            generated_image: 生成的彩色图像 (B, 3, H, W)。

        Returns:
            标量损失。
        """
        real_features = dino_encoder(real_image)
        generated_features = dino_encoder(generated_image)
        height, width = real_features[0].shape[-2:]

        patch_ids = np.random.permutation(height * width)[: min(self.num_patches, height * width)]
        patch_ids = torch.tensor(patch_ids, dtype=torch.long, device=real_image.device)
        keys = self._project(real_features, patch_ids)              # 真实特征不回传梯度
        queries = self._project(generated_features, patch_ids)

        total_loss = 0.0
        for query, key in zip(queries, keys):
            key = key.detach()
            positive = (query * key).sum(dim=1, keepdim=True)       # (N, 1) 正样本相似度
            negative = query @ key.t()                              # (N, N) 负样本相似度
            negative.fill_diagonal_(-10.0)                          # 自身不算负样本
            positive_exp = torch.exp(positive / self.temperature)
            negative_exp = torch.exp(negative / self.temperature)
            ratio = positive_exp / (positive_exp + negative_exp.sum(dim=1, keepdim=True))
            total_loss = total_loss - torch.log(ratio).mean()
        return total_loss * self.lambda_nce / self.num_layers


class MMDLoss(nn.Module):
    """多层特征最大均值差异损失。

    用 RBF 核度量真实光学图像与生成图像在 DINOv2 多层特征上的空间块分布差异，
    缓解整体色调偏移、提升色彩多样性。

    Args:
        lambda_mmd: 损失权重。
        sigma: RBF 核带宽。
    """

    def __init__(self, lambda_mmd=5.0, sigma=1.0):
        super().__init__()
        self.lambda_mmd = lambda_mmd
        self.sigma = sigma

    def _gaussian_kernel(self, x, y):
        """计算 (N, D) 与 (M, D) 特征之间的 RBF 核矩阵 (N, M)。"""
        squared_distance = ((x.unsqueeze(1) - y.unsqueeze(0)) ** 2).sum(dim=2)
        return torch.exp(-squared_distance / (2 * self.sigma ** 2))

    def forward(self, dino_encoder, real_image, generated_image):
        """计算 MMD 损失。

        Args:
            dino_encoder: 冻结的 DINOv2 编码器。
            real_image: 真实光学图像 (B, 3, H, W)，取值 [-1, 1]。
            generated_image: 生成的彩色图像 (B, 3, H, W)。

        Returns:
            标量损失。
        """
        total_mmd = torch.zeros((), device=real_image.device)
        for real_feat, gen_feat in zip(dino_encoder(real_image), dino_encoder(generated_image)):
            channels = real_feat.shape[1]
            x = real_feat.permute(0, 2, 3, 1).reshape(-1, channels)     # (B*H*W, C)
            y = gen_feat.permute(0, 2, 3, 1).reshape(-1, channels)
            k_xx = self._gaussian_kernel(x, x)
            k_yy = self._gaussian_kernel(y, y)
            k_xy = self._gaussian_kernel(x, y)
            total_mmd = total_mmd + k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()
        return total_mmd * self.lambda_mmd
