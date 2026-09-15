"""配置模块：集中定义训练与推理的默认参数。

默认值即默认实验配置，命令行参数可覆盖。
"""

import argparse
import os

dataset_dict= {
    'sar2opt': r'C:\M\datasets\sar2opt\n4000',  # 替换为自己的数据集路径
}

def build_train_parser():
    """构建训练参数解析器。"""
    parser = argparse.ArgumentParser(description="微调单步扩散模型完成 SAR 灰度图 -> 光学彩色图翻译 (b2a)。")

    parser.add_argument("--dataset_name", type=str, default="sar2opt", help="配置文件路径，优先级高于命令行参数")

    # 运行环境
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda", help="运算设备")

    # 数据集
    parser.add_argument("--train_img_prep", type=str, default="no_resize", help="训练图像预处理策略")
    parser.add_argument("--val_img_prep", type=str, default="no_resize", help="验证图像预处理策略")
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--max_train_epochs", type=int, default=200)
    parser.add_argument("--max_train_steps", type=int, default=25000)

    # 模型
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/sd-turbo")
    parser.add_argument("--lora_rank_unet", type=int, default=16, help="UNet LoRA 秩")
    parser.add_argument("--lora_rank_vae", type=int, default=4, help="VAE LoRA 秩")

    # 损失权重
    parser.add_argument("--lambda_gan", type=float, default=0.5)
    parser.add_argument("--lambda_idt", type=float, default=0.0)
    parser.add_argument("--lambda_idt_lpips", type=float, default=0.0)
    parser.add_argument("--lambda_NCE", type=float, default=0.2, help="PatchNCE 对比损失权重")
    parser.add_argument("--lambda_MMD", type=float, default=1.0, help="MMD 分布对齐损失权重")

    # DINO 特征
    parser.add_argument("--dino_model", type=str, default="dinov2_vits14")
    parser.add_argument("--dino_patch_size", type=int, default=8, help="patch 化步长，256 输入得到 32x32 特征图")
    parser.add_argument("--feat_layers", type=str, default="5,11", help="参与损失的特征层，逗号分隔")
    parser.add_argument("--nce_num_patches", type=int, default=64, help="对比损失采样的图像块数量")
    parser.add_argument("--nce_temperature", type=float, default=0.07)

    # 优化器
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=10.0)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    # 验证与保存
    parser.add_argument("--validation_steps", type=int, default=1000, help="验证间隔步数")
    parser.add_argument("--validation_num_images", type=int, default=-1, help="验证图像数，-1 表示全部")
    parser.add_argument("--checkpointing_steps", type=int, default=5000, help="权重保存间隔步数")

    # 显存优化
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--xformers", action=argparse.BooleanOptionalAction, default=True)
    return parser


def resolve_train_config(argv=None):
    """解析训练配置，并补全 run_name / run_dir / dataset_name / feat_layers。"""
    config = build_train_parser().parse_args(argv)
    config.dataset_folder = dataset_dict.get(config.dataset_name, "")

    config.feat_layers = [int(layer) for layer in config.feat_layers.split(",") if layer]

    config.project_name = f'runs_{config.dataset_name}'
    config.run_name = f"CFDM-nce{config.lambda_NCE}-mmd{config.lambda_MMD}"

    config.run_dir = os.path.join(config.project_name, config.run_name)
    return config


def build_infer_parser():
    """构建推理参数解析器。"""
    parser = argparse.ArgumentParser(description="用微调权重把 SAR 灰度图彩色化为光学风格图像 (b2a)。")
    parser.add_argument("--checkpoint", type=str, required=True, help="微调权重 .pkl 路径")
    parser.add_argument("--input", type=str, required=True, help="输入：单张图像或文件夹")
    parser.add_argument("--output_dir", type=str, default="results", help="结果保存目录")
    parser.add_argument("--prompt", type=str, default="an optical remote sensing image", help="目标域提示词")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resize", type=int, default=0, help="统一缩放到 NxN，0 表示保持原尺寸")
    parser.add_argument("--save_comparison", action="store_true", default=True, help="额外保存「输入|输出」拼接图")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/sd-turbo")
    return parser


def resolve_infer_config(argv=None):
    """解析推理配置。"""
    return build_infer_parser().parse_args(argv)
