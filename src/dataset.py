"""非配对图像翻译数据集。

数据目录结构：
    <dataset_folder>/
    ├── fixed_prompt.txt   # 固定提示词
    ├── trainA/ trainB/      # 训练集，两域图像无需一一对应
    └── testA/  testB/       # 测试集，文件名一一对应时可算 PSNR/SSIM/LPIPS
"""

import os
import random

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms

from .utils import list_images


def build_transform(image_prep):
    """按名称返回图像预处理流水线。"""
    pipelines = {
        "no_resize": [],
        "resize_256": [transforms.Resize((256, 256), interpolation=Image.LANCZOS)],
        "resize_512": [transforms.Resize((512, 512), interpolation=Image.LANCZOS)],
        "resize_286_randomcrop_256x256_hflip": [
            transforms.Resize((286, 286), interpolation=Image.LANCZOS),
            transforms.RandomCrop((256, 256)),
            transforms.RandomHorizontalFlip(),
        ],
    }
    if image_prep not in pipelines:
        raise ValueError(f"不支持的图像预处理: {image_prep}，可选 {list(pipelines)}")
    return transforms.Compose(pipelines[image_prep])


class UnpairedDataset(torch.utils.data.Dataset):
    """非配对数据集：A 域按索引顺序取图，B 域随机取图。

    Args:
        dataset_folder: 数据集根目录。
        split: "train" 或 "test"。
        image_prep: 预处理策略名，见 build_transform。
        tokenizer: CLIP 分词器，用于编码固定提示词。
    """

    def __init__(self, dataset_folder, split, image_prep, tokenizer):
        if split not in ("train", "test"):
            raise ValueError(f"split 必须为 train 或 test，收到 {split}")
        self.source_images = list_images(os.path.join(dataset_folder, f"{split}A"))
        self.target_images = list_images(os.path.join(dataset_folder, f"{split}B"))
        if not self.source_images or not self.target_images:
            raise FileNotFoundError(f"{dataset_folder} 下的 {split}A/{split}B 目录没有图像")

        with open(os.path.join(dataset_folder, "fixed_prompt.txt"), encoding="utf-8") as file:
            self.fixed_caption_src = file.read().strip()

        self.tokenizer = tokenizer
        self.transform = build_transform(image_prep)
        self.input_ids_src = self._tokenize(self.fixed_caption_src)

    def _tokenize(self, caption):
        """把提示词编码为 (1, 77) 的 token id。"""
        return self.tokenizer(
            caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids

    def __len__(self):
        return len(self.source_images) + len(self.target_images)

    def __getitem__(self, index):
        """返回两个域各一张图像（都归一化到 [-1, 1]）与各自的固定提示词。"""
        source_path = self.source_images[index] if index < len(self.source_images) \
            else random.choice(self.source_images)
        target_path = random.choice(self.target_images)

        source = TF.to_tensor(self.transform(Image.open(source_path).convert("RGB")))
        target = TF.to_tensor(self.transform(Image.open(target_path).convert("RGB")))
        return {
            "pixel_values_src": TF.normalize(source, mean=[0.5], std=[0.5]),
            "pixel_values_tgt": TF.normalize(target, mean=[0.5], std=[0.5]),
            "caption_src": self.fixed_caption_src,
            "input_ids_src": self.input_ids_src,
            "input_ids_tgt": self.input_ids_tgt,
        }
