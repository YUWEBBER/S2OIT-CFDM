"""通用工具：日志、随机种子、图像路径与张量读写。"""

import glob
import logging
import os

import numpy as np
import torch
from PIL import Image


def ensure_dir(path):
    """创建目录（含父目录），返回原路径。"""
    if path:
        os.makedirs(path, exist_ok=True)
    return path


def setup_logger(name):
    """创建输出到控制台的日志器。"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def seed_everything(seed):
    """设置随机种子，保证实验可复现。"""
    from accelerate.utils import set_seed

    set_seed(int(seed))


def resolve_device(device):
    """解析运算设备，CUDA 不可用时回退到 CPU。"""
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        logging.getLogger("cfdm").warning("CUDA 不可用，回退到 CPU")
        return torch.device("cpu")
    return torch.device(device)


def list_images(folder):
    """返回目录下 .jpg / .png 图像的排序路径列表。"""
    if not os.path.isdir(folder):
        return []
    paths = [p for p in glob.glob(os.path.join(folder, "*")) if p.lower().endswith((".jpg", ".png"))]
    return sorted(paths)


def tensor_to_pil(tensor):
    """把 [-1, 1] 区间的图像张量 (3, H, W) 转为 8 位 PIL 图像。"""
    array = (tensor.detach().float().cpu() * 0.5 + 0.5).clamp(0, 1) * 255
    array = array.permute(1, 2, 0).numpy().round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")
