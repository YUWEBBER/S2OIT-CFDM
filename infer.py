"""推理入口：用微调权重把 SAR 灰度图彩色化为光学风格图像（b2a）。

支持单张图像或整个文件夹，结果按原文件名保存为 PNG；可选输出「输入|输出」拼接图，
以及以 gt_dir 为参考计算评价指标。

用法：python infer.py --checkpoint checkpoints/model_5001.pkl --input data/sar2opt/testB
"""

import os
import time

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

from src.config import resolve_infer_config
from src.metrics import evaluate_image_folders
from src.model import CFDM
from src.utils import ensure_dir, list_images, resolve_device, setup_logger, tensor_to_pil

SIZE_MULTIPLE = 8  # 模型要求输入边长是 8 的整数倍
NORMALIZE = transforms.Normalize([0.5], [0.5])


def main():
    config = resolve_infer_config()
    device = resolve_device(config.device)
    ensure_dir(config.output_dir)
    logger = setup_logger("cfdm.infer")

    if not os.path.exists(config.checkpoint):
        raise FileNotFoundError(f"权重文件不存在: {config.checkpoint}")
    logger.info("加载模型（首次运行会自动下载基础模型）...")
    model = CFDM(
        pretrained_path=config.checkpoint,
        pretrained_name=config.pretrained_model_name_or_path,
        device=device,
    )
    model.eval()

    if os.path.isfile(config.input):
        image_paths = [config.input]
    else:
        image_paths = list_images(config.input)
    if not image_paths:
        raise FileNotFoundError(f"未在 {config.input} 中找到图像")
    logger.info("待处理图像: %d 张", len(image_paths))

    # 目标域（光学）文本嵌入只需计算一次
    token_ids = model.tokenizer(
        config.prompt, max_length=model.tokenizer.model_max_length,
        padding="max_length", truncation=True, return_tensors="pt",
    ).input_ids.to(device)
    caption_emb = model.text_encoder(token_ids)[0].detach()

    output_abs = os.path.abspath(config.output_dir)
    comparison_dir = os.path.join(os.path.dirname(output_abs), os.path.basename(output_abs) + "_comparison")
    if config.save_comparison:
        ensure_dir(comparison_dir)

    started = time.time()
    for image_path in tqdm(image_paths, desc="Colorizing"):
        source = Image.open(image_path).convert("RGB")
        target_size = source.size                       # (W, H)，用于裁剪回原分辨率
        if config.resize > 0:
            source = source.resize((config.resize, config.resize), Image.LANCZOS)
            target_size = (config.resize, config.resize)

        # 归一化到 [-1, 1]，并把边长补齐到 8 的整数倍
        inputs = NORMALIZE(transforms.ToTensor()(source)).unsqueeze(0)
        pad_width = -inputs.shape[3] % SIZE_MULTIPLE
        pad_height = -inputs.shape[2] % SIZE_MULTIPLE
        if pad_width or pad_height:
            inputs = F.pad(inputs, (0, pad_width, 0, pad_height), mode="reflect")
        inputs = inputs.to(device)

        with torch.no_grad():
            generated = model.translate(inputs, caption_emb)
        result = tensor_to_pil(generated[0][:, : target_size[1], : target_size[0]])

        output_path = os.path.join(
            config.output_dir, os.path.splitext(os.path.basename(image_path))[0] + ".png"
        )
        result.save(output_path)
        if config.save_comparison:
            # 拼接图放在结果目录的同级目录，避免被 FID/KID 当成生成结果统计
            canvas = Image.new("RGB", (source.width * 2 + 8, source.height), (255, 255, 255))
            canvas.paste(source, (0, 0))
            canvas.paste(result, (source.width + 8, 0))
            canvas.save(os.path.join(comparison_dir, os.path.splitext(os.path.basename(image_path))[0] + ".jpg"),
                        quality=95)

    logger.info("完成 %d 张，用时 %.2f 秒", len(image_paths), time.time() - started)
    if config.save_comparison:
        logger.info("拼接对比图目录: %s", os.path.abspath(comparison_dir))


if __name__ == "__main__":
    main()
