"""图像质量评价指标：LPIPS / FID / KID / DINO-Struct。

LPIPS / DINO-Struct 按图像主文件名（不含扩展名）配对，需要参考目录与结果
目录中有同名图像；FID / KID 为分布距离，不要求配对。各指标越小越好。
"""

import logging
import os

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from .dino import DinoStructureScore
from .utils import ensure_dir, list_images

PRECISION = 4


def _matched_pairs(reference_dir, generated_dir):
    """按主文件名配对参考图像与生成图像。
    Returns:
        [(参考图像路径, 生成图像路径)]，缺失配对会被跳过并告警。
    """
    reference_paths = list_images(reference_dir)
    generated = {os.path.splitext(os.path.basename(p))[0]: p for p in list_images(generated_dir)}
    pairs = []
    for path in reference_paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem in generated:
            pairs.append((path, generated[stem]))
    if len(pairs) < len(reference_paths):
        logging.getLogger("cfdm").warning("%d 张参考图像缺少对应生成结果，已跳过", len(reference_paths) - len(pairs))
    return pairs


def compute_lpips(reference_dir, generated_dir, device="cuda"):
    """配对图像的 LPIPS 均值。"""
    import lpips

    loss_fn = lpips.LPIPS(net="vgg").to(device).eval()
    scores = []
    for reference_path, generated_path in tqdm(_matched_pairs(reference_dir, generated_dir), desc="LPIPS"):
        with torch.no_grad():
            reference = lpips.im2tensor(lpips.load_image(reference_path)).to(device)
            generated = lpips.im2tensor(lpips.load_image(generated_path)).to(device)
            scores.append(float(loss_fn(reference, generated).item()))
    return round(float(np.mean(scores)), PRECISION)


def compute_dino_structure(reference_dir, generated_dir, device="cuda"):
    """配对图像的 DINO-Struct 结构一致性均值。"""
    scorer = DinoStructureScore(device=device)
    scores = []
    for reference_path, generated_path in tqdm(_matched_pairs(reference_dir, generated_dir), desc="DINO-Struct"):
        reference = Image.open(reference_path).convert("RGB")
        generated = Image.open(generated_path).convert("RGB")
        scores.append(float(scorer(reference, generated).item()))
    return round(float(np.mean(scores)), PRECISION)


def _kid(reference_features, generated_features, n_subsets=50, subset_size=1000):
    """KID：多项式核无偏 MMD^2 在随机子集上的均值。
    Args:
        reference_features / generated_features: (N, D) 的 Inception 特征。
        n_subsets: 子集采样次数。
        subset_size: 每次采样的样本数，超过图像总数时自动截断。
    Returns:
        KID 标量。
    """
    from sklearn.metrics.pairwise import polynomial_kernel

    subset_size = min(subset_size, len(reference_features), len(generated_features))
    m = subset_size
    mmds = np.zeros(n_subsets)
    for index in tqdm(range(n_subsets), desc="KID", leave=False):
        reference = reference_features[np.random.choice(len(reference_features), m, replace=False)]
        generated = generated_features[np.random.choice(len(generated_features), m, replace=False)]
        # 无偏估计：核矩阵去掉对角线后除以 m(m-1)，交叉项除以 m^2
        k_xx = polynomial_kernel(reference, degree=3, coef0=1)
        k_yy = polynomial_kernel(generated, degree=3, coef0=1)
        k_xy = polynomial_kernel(reference, generated, degree=3, coef0=1)
        sum_xx = (k_xx.sum() - np.diagonal(k_xx).sum()) / (m * (m - 1))
        sum_yy = (k_yy.sum() - np.diagonal(k_yy).sum()) / (m * (m - 1))
        mmds[index] = sum_xx + sum_yy - 2 * k_xy.sum() / (m * m)
    return float(mmds.mean() * 100)


def _folder_features(folder, feature_model, device):
    """用 clean-fid 的 Inception 提取目录内全部图像的特征，返回 (N, 2048)。"""
    from cleanfid.fid import get_folder_features

    return get_folder_features(
        folder, model=feature_model, num_workers=0, num=None, shuffle=False, seed=0, batch_size=8,
        device=torch.device(device), mode="clean", custom_fn_resize=None, description="",
        verbose=True, custom_image_tranform=None,
    )


def compute_fid_kid(reference_dir, generated_dir, device="cuda"):
    """FID 与 KID"""
    from cleanfid.fid import build_feature_extractor, frechet_distance

    feature_model = build_feature_extractor("clean", str(device), use_dataparallel=False)
    reference_features = _folder_features(reference_dir, feature_model, device)
    generated_features = _folder_features(generated_dir, feature_model, device)
    fid = frechet_distance(
        np.mean(reference_features, axis=0), np.cov(reference_features, rowvar=False),
        np.mean(generated_features, axis=0), np.cov(generated_features, rowvar=False),
    )
    return round(float(fid), PRECISION), round(_kid(reference_features, generated_features), PRECISION)


def evaluate_image_folders(reference_dir, generated_dir, input_dir, device="cuda", report_path=None):
    """计算全部指标。

    Args:
        reference_dir: 参考（真实光学）图像目录。
        generated_dir: 生成结果目录。
        input_dir: 输入（SAR）图像目录。
        device: 运算设备。
        report_path: 指标报告 .txt 输出路径，None 表示不写文件。

    Returns:
        {"lpips":.., "fid":.., "kid":.., "dino":..}
    """
    metrics = {}
    metrics["lpips"] = compute_lpips(reference_dir, generated_dir, device)
    metrics["fid"], metrics["kid"] = compute_fid_kid(reference_dir, generated_dir, device)
    metrics["dino"] = compute_dino_structure(input_dir, generated_dir, device)

    if report_path:
        ensure_dir(os.path.dirname(os.path.abspath(report_path)))
        lines = [f"# reference: {os.path.abspath(reference_dir)}",
                 f"# generated: {os.path.abspath(generated_dir)}"]
        lines += [f"{key}: {value}" for key, value in metrics.items()]
        with open(report_path, "w", encoding="utf-8") as file:
            file.write("\n".join(lines) + "\n")
    return metrics
