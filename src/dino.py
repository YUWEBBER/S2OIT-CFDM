"""DINO 相关实现。

- build_dino_feature_extractor: 训练用 DINOv2 多层特征提取器（供对比/MMD 损失使用）。
- DinoStructureScore: 评价用 DINO-Struct 结构一致性指标。

首次运行会通过 Torch Hub 自动下载 DINOv2 / DINO 预训练权重。
"""

import torch
import torch.nn.functional as F
import torchvision


def build_dino_feature_extractor(model_name="dinov2_vits14", feat_layers=(5, 11), patch_size=8, device="cuda"):
    """构建冻结的 DINOv2 多层特征提取器。

    只覆盖 patch 嵌入的卷积步长与填充（卷积核保持骨干原生的 14x14），使 256x256 输入
    得到 32x32 特征图，从而在更细的空间粒度上计算损失。

    Returns:
        DINOv2 模型，forward(images) 返回特征图列表，每层形状 (B, embed_dim, H/ps, W/ps)。
    """
    encoder = torch.hub.load("facebookresearch/dinov2", model_name, verbose=False)
    embed_dim = int(encoder.embed_dim)
    feature_layers = list(feat_layers)

    encoder.patch_embed.proj.stride = (patch_size, patch_size)
    encoder.patch_embed.proj.padding = (6, 6)
    encoder.patch_size = patch_size
    encoder.patch_embed.patch_size = (patch_size, patch_size)

    def forward(self, images):
        batch, _, height, width = images.shape
        assert height % patch_size == 0 and width % patch_size == 0, f"输入边长需为 {patch_size} 的整数倍"

        tokens = self.prepare_tokens_with_masks(images, masks=None)     # (B, 1 + N, C)
        feat_size = int(round((tokens.shape[1] - 1) ** 0.5))
        features = []
        for index, block in enumerate(self.blocks):
            tokens = block(tokens)
            if index in feature_layers:
                # 去掉 cls token，还原为 (B, C, feat_size, feat_size)
                features.append(tokens[:, 1:, :].transpose(2, 1).view(batch, embed_dim, feat_size, feat_size))
        return features

    encoder.forward = forward.__get__(encoder, encoder.__class__)
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    encoder.eval()
    return encoder.to(device)


class DinoStructureScore:
    """DINO-Struct 结构一致性指标。

    比较参考图像与生成图像在 DINO ViT 第 11 层注意力键自相似矩阵上的 MSE，数值越小结构越一致。

    Args:
        device: 运算设备。
        model_name: DINO 模型名。
    """

    LAYER = 11

    def __init__(self, device="cuda", model_name="dino_vitb8"):
        self.device = torch.device(device)
        self.model = torch.hub.load("facebookresearch/dino:main", model_name, verbose=False)
        self.model = self.model.to(self.device).eval()
        self.preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def _key_self_similarity(self, image):
        """抓取第 11 层 qkv，用 key 分支计算 token 之间的余弦相似度矩阵 (N, N)。"""
        captured = {}

        def hook(_module, _inputs, output):
            captured["qkv"] = output

        handle = self.model.blocks[self.LAYER].attn.qkv.register_forward_hook(hook)
        self.model(image)
        handle.remove()

        attention = self.model.blocks[self.LAYER].attn
        num_heads = attention.num_heads
        qkv = captured["qkv"].reshape(-1, captured["qkv"].shape[-1])       # (N, 3*C)
        head_dim = qkv.shape[-1] // 3 // num_heads
        keys = qkv.reshape(qkv.shape[0], 3, num_heads, head_dim)
        keys = keys.permute(1, 2, 0, 3)[1]                                 # (heads, N, head_dim)
        keys = keys.permute(1, 0, 2).reshape(qkv.shape[0], num_heads * head_dim)   # (N, C)
        norm = keys.norm(dim=1, keepdim=True)                              # (N, 1)
        return (keys @ keys.t()) / (norm @ norm.t()).clamp(min=1e-8)

    @torch.no_grad()
    def __call__(self, reference, generated):
        """计算单张图像的 DINO-Struct 得分。

        Args:
            reference: 参考（光学真值）PIL 图像。
            generated: 生成 PIL 图像。

        Returns:
            标量张量。
        """
        target = self._key_self_similarity(self.preprocess(reference).unsqueeze(0).to(self.device))
        value = self._key_self_similarity(self.preprocess(generated).unsqueeze(0).to(self.device))
        return F.mse_loss(value, target)
