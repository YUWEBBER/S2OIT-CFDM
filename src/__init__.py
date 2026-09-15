"""SAR 图像彩色化：基于微调单步扩散模型把 SAR 灰度图翻译为光学彩色图。

模块说明：
    config    训练/推理默认参数与命令行解析
    dataset   非配对数据集与图像预处理
    model     调度器、VAE 前向改写、LoRA 注入与 CycleGAN-Turbo 模型
    losses    PatchNCE 对比损失与 MMD 分布对齐损失
    dino      DINOv2 特征提取器与 DINO-Struct 结构指标
    metrics   PSNR / SSIM / LPIPS / FID / KID / DINO-Struct 评价
    utils     日志、随机种子、图像路径与张量读写
"""

__version__ = "1.0.0"
