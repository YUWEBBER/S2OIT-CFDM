"""训练入口：微调单步扩散模型完成 SAR 灰度图 -> 光学彩色图翻译。

只训练 LoRA 适配器与 VAE skip 卷积，主干冻结；每个损失单独 backward 与优化器 step。
训练日志与指标写入 swanlab，run_dir 下另存配置快照与权重。

用法：python train.py --dataset_folder data/sar2opt
"""

import gc
import json
import os
import shutil

import lpips
import torch
import torch.nn.functional as F
import vision_aided_loss
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from accelerate import Accelerator
from diffusers.optimization import get_scheduler
from peft.utils import get_peft_model_state_dict
from transformers import AutoTokenizer, CLIPTextModel

from src.config import resolve_train_config
from src.dataset import UnpairedDataset, build_transform
from src.dino import build_dino_feature_extractor
from src.losses import MMDLoss, NCELoss
from src.metrics import evaluate_image_folders
from src.model import (
    CFDM,
    VAE_decode,
    VAE_encode,
    initialize_unet,
    initialize_vae,
    make_1step_sched,
)
from src.utils import ensure_dir, list_images, seed_everything, setup_logger, tensor_to_pil


def train(config):
    """执行训练流程。"""
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="swanlab",
        project_dir=config.project_name,
    )
    seed_everything(config.seed)
    device = accelerator.device
    logger = setup_logger("cfdm.train")

    if accelerator.is_main_process:
        ensure_dir(config.run_dir)
        # 保存配置快照与训练脚本，便于复现
        with open(os.path.join(config.run_dir, "config.json"), "w", encoding="utf-8") as file:
            json.dump(vars(config), file, indent=2, ensure_ascii=False)
        shutil.copyfile(__file__, os.path.join(config.run_dir, "train.py"))
    logger.info("实验目录: %s", os.path.abspath(config.run_dir))

    # ---------- 基础模型（CLIP 文本编码器 / UNet / VAE / 调度器） ----------
    tokenizer = AutoTokenizer.from_pretrained(
        config.pretrained_model_name_or_path, subfolder="tokenizer", use_fast=False
    )
    text_encoder = CLIPTextModel.from_pretrained(
        config.pretrained_model_name_or_path, subfolder="text_encoder"
    )
    scheduler = make_1step_sched(config.pretrained_model_name_or_path, device)
    unet, unet_module_names = initialize_unet(config.lora_rank_unet, config.pretrained_model_name_or_path)
    vae, vae_lora_modules = initialize_vae(config.lora_rank_vae, config.pretrained_model_name_or_path, device)

    text_encoder.to(device)
    text_encoder.requires_grad_(False)
    unet.to(device, dtype=torch.float32)
    vae.to(device, dtype=torch.float32)

    if config.xformers:
        unet.enable_xformers_memory_efficient_attention()
    if config.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    if config.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # ---------- 判别器与损失 ----------
    # 视觉辅助判别器：用冻结的 CLIP 视觉主干做特征，只训练判别头
    net_disc = vision_aided_loss.Discriminator(cv_type="clip", loss_type="multilevel_sigmoid", device=str(device))
    net_disc.cv_ensemble.requires_grad_(False)

    dino_encoder = build_dino_feature_extractor(
        config.dino_model, config.feat_layers, config.dino_patch_size, device
    )
    contrast_loss = NCELoss(
        feature_dim=int(dino_encoder.embed_dim),
        num_layers=len(config.feat_layers),
        num_patches=config.nce_num_patches,
        lambda_nce=config.lambda_NCE,
        temperature=config.nce_temperature,
        device=device,
    )
    mmd_loss = MMDLoss(lambda_mmd=config.lambda_MMD)
    net_lpips = lpips.LPIPS(net="vgg").to(device)
    net_lpips.requires_grad_(False)

    # ---------- 数据与优化器 ----------
    dataset_train = UnpairedDataset(config.dataset_folder, "train", config.train_img_prep, tokenizer)
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train, batch_size=config.train_batch_size, shuffle=True,
        num_workers=config.dataloader_num_workers,
    )
    logger.info("训练样本数: %d | 数据集: %s", len(dataset_train), config.dataset_folder)

    # b2a 方向生成的是光学图像，因此使用 A 域（光学）提示词
    token_ids = tokenizer(
        dataset_train.fixed_caption_src, max_length=tokenizer.model_max_length,
        padding="max_length", truncation=True, return_tensors="pt",
    ).input_ids.to(device)
    fixed_optical_emb = text_encoder(token_ids)[0].detach()          # (1, 77, 1024)
    del text_encoder, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    vae_enc, vae_dec = VAE_encode(vae), VAE_decode(vae)
    params_gen = CFDM.get_trainable_params(unet, vae)
    params_disc = list(net_disc.parameters())
    optimizer_gen = torch.optim.AdamW(
        params_gen, lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay, eps=config.adam_epsilon,
    )
    optimizer_disc = torch.optim.AdamW(
        params_disc, lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay, eps=config.adam_epsilon,
    )
    optimizer_projector = torch.optim.AdamW(
        contrast_loss.projectors.parameters(), lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay, eps=config.adam_epsilon,
    )
    warmup_steps = config.lr_warmup_steps * accelerator.num_processes
    total_steps = config.max_train_steps * accelerator.num_processes
    lr_scheduler_gen = get_scheduler(
        config.lr_scheduler, optimizer_gen, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    lr_scheduler_disc = get_scheduler(
        config.lr_scheduler, optimizer_disc, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    unet, vae_enc, vae_dec, net_disc = accelerator.prepare(unet, vae_enc, vae_dec, net_disc)
    (net_lpips, optimizer_gen, optimizer_disc, optimizer_projector, train_dataloader,
     lr_scheduler_gen, lr_scheduler_disc) = accelerator.prepare(
        net_lpips, optimizer_gen, optimizer_disc, optimizer_projector, train_dataloader,
        lr_scheduler_gen, lr_scheduler_disc,
    )
    # 关闭判别器主干中的融合注意力，避免与 xformers 版本的数值差异
    for name, module in net_disc.named_modules():
        if "attn" in name:
            module.fused_attn = False

    if accelerator.is_main_process:
        accelerator.init_trackers(
            config.dataset_name, config=vars(config),
            init_kwargs={"swanlab": {"experiment_name": config.run_name}},
        )
    logger.info("开始训练：epochs=%d, max_steps=%d, batch=%d",
                config.max_train_epochs, config.max_train_steps, config.train_batch_size)

    val_transform = build_transform(config.val_img_prep)
    global_step = 0
    progress_bar = tqdm(
        range(config.max_train_steps), initial=0, desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for _epoch in range(config.max_train_epochs):
        for batch in train_dataloader:
            if global_step >= config.max_train_steps:
                break

            with accelerator.accumulate(unet, net_disc, vae_enc, vae_dec):
                img_optical = batch["pixel_values_src"].to(dtype=torch.float32)   # 域 A：光学参考
                img_sar = batch["pixel_values_tgt"].to(dtype=torch.float32)       # 域 B：SAR 输入
                batch_size = img_sar.shape[0]
                text_emb = fixed_optical_emb.repeat(batch_size, 1, 1)
                timesteps = torch.full(
                    (batch_size,), scheduler.config.num_train_timesteps - 1,
                    device=device, dtype=torch.long,
                )

                # (1) 生成器对抗损失
                fake_optical = CFDM.forward_with_networks(
                    img_sar, vae_enc, unet, vae_dec, scheduler, timesteps, text_emb
                )
                loss_gan = net_disc(fake_optical, for_G=True).mean() * config.lambda_gan
                accelerator.backward(loss_gan)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_gen, config.max_grad_norm)
                optimizer_gen.step()
                lr_scheduler_gen.step()
                optimizer_gen.zero_grad()
                optimizer_disc.zero_grad()

                # (2) 生成器恒等损失（默认关闭）
                if config.lambda_idt > 0:
                    identity_optical = CFDM.forward_with_networks(
                        img_optical, vae_enc, unet, vae_dec, scheduler, timesteps, text_emb
                    )
                    loss_idt = F.l1_loss(identity_optical, img_optical) * config.lambda_idt
                    if config.lambda_idt_lpips > 0:
                        loss_idt = loss_idt + net_lpips(identity_optical, img_optical).mean() * config.lambda_idt_lpips
                    accelerator.backward(loss_idt)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(params_gen, config.max_grad_norm)
                    optimizer_gen.step()
                    lr_scheduler_gen.step()
                    optimizer_gen.zero_grad()
                else:
                    loss_idt = torch.zeros_like(loss_gan)

                # (3) 生成器对比损失：约束 SAR 与生成彩色图的结构对应
                if config.lambda_NCE > 0:
                    fake_optical = CFDM.forward_with_networks(
                        img_sar, vae_enc, unet, vae_dec, scheduler, timesteps, text_emb
                    )
                    loss_nce = contrast_loss(dino_encoder, img_sar, fake_optical)
                    accelerator.backward(loss_nce)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(params_gen, config.max_grad_norm)
                    optimizer_gen.step()
                    lr_scheduler_gen.step()
                    optimizer_gen.zero_grad()
                    optimizer_projector.step()
                    optimizer_projector.zero_grad()
                else:
                    loss_nce = torch.zeros_like(loss_gan)

                # (4) 生成器 MMD 损失：对齐真实光学与生成彩色图的特征分布
                if config.lambda_MMD > 0:
                    fake_optical = CFDM.forward_with_networks(
                        img_sar, vae_enc, unet, vae_dec, scheduler, timesteps, text_emb
                    )
                    loss_mmd = mmd_loss(dino_encoder, img_optical, fake_optical)
                    accelerator.backward(loss_mmd)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(params_gen, config.max_grad_norm)
                    optimizer_gen.step()
                    lr_scheduler_gen.step()
                    optimizer_gen.zero_grad()
                else:
                    loss_mmd = torch.zeros_like(loss_gan)

                # (5) 判别器：假样本与真样本各更新一次
                loss_disc_fake = net_disc(fake_optical.detach(), for_real=False).mean() * config.lambda_gan
                accelerator.backward(loss_disc_fake)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_disc, config.max_grad_norm)
                optimizer_disc.step()
                lr_scheduler_disc.step()
                optimizer_disc.zero_grad()

                loss_disc_real = net_disc(img_optical, for_real=True).mean() * config.lambda_gan
                accelerator.backward(loss_disc_real)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_disc, config.max_grad_norm)
                optimizer_disc.step()
                lr_scheduler_disc.step()
                optimizer_disc.zero_grad()

            logs = {
                "loss_gan": float(loss_gan.detach().item()),
                "loss_disc": float((loss_disc_fake + loss_disc_real).detach().item()),
                "loss_nce": float(loss_nce.detach().item()),
                "loss_mmd": float(loss_mmd.detach().item()),
                "lr": float(optimizer_gen.param_groups[0]["lr"]),
            }
            if config.lambda_idt > 0:
                logs["loss_idt"] = float(loss_idt.detach().item())

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            # 断点保存与验证：首个断点与首次验证在第 N+1 步（与原实现一致）
            if accelerator.is_main_process and accelerator.sync_gradients:
                if global_step % config.checkpointing_steps == 0 and global_step > 2:
                    checkpoint_path = os.path.join(config.run_dir, "checkpoints", f"model_{global_step}.pkl")
                    ensure_dir(os.path.dirname(checkpoint_path))
                    eval_unet = accelerator.unwrap_model(unet)
                    eval_vae_enc = accelerator.unwrap_model(vae_enc)
                    encoder_modules, decoder_modules, other_modules = unet_module_names
                    torch.save({
                        "l_target_modules_encoder": encoder_modules,
                        "l_target_modules_decoder": decoder_modules,
                        "l_modules_others": other_modules,
                        "rank_unet": config.lora_rank_unet,
                        "sd_encoder": get_peft_model_state_dict(eval_unet, adapter_name="default_encoder"),
                        "sd_decoder": get_peft_model_state_dict(eval_unet, adapter_name="default_decoder"),
                        "sd_other": get_peft_model_state_dict(eval_unet, adapter_name="default_others"),
                        "rank_vae": config.lora_rank_vae,
                        "vae_lora_target_modules": vae_lora_modules,
                        "sd_vae_enc": eval_vae_enc.state_dict(),   # 编码器与解码器共享权重，只存一份
                    }, checkpoint_path)
                    logger.info("已保存权重: %s", checkpoint_path)
                    gc.collect()
                    torch.cuda.empty_cache()

                if global_step % config.validation_steps == 0:
                    sample_dir = os.path.join(config.run_dir, f"validation-{global_step}", "samples_gen")
                    ensure_dir(sample_dir)
                    val_images = list_images(os.path.join(config.dataset_folder, "testB"))
                    if config.validation_num_images > 0:
                        val_images = val_images[: config.validation_num_images]

                    eval_unet = accelerator.unwrap_model(unet)
                    eval_vae_enc = accelerator.unwrap_model(vae_enc)
                    eval_vae_dec = accelerator.unwrap_model(vae_dec)
                    val_timesteps = torch.full(
                        (1,), scheduler.config.num_train_timesteps - 1, device=device, dtype=torch.long
                    )
                    for image_path in tqdm(val_images, desc="validate"):
                        image = val_transform(Image.open(image_path).convert("RGB"))
                        inputs = transforms.Normalize([0.5], [0.5])(
                            transforms.ToTensor()(image)
                        ).unsqueeze(0).to(device)
                        generated = CFDM.forward_with_networks(
                            inputs, eval_vae_enc, eval_unet, eval_vae_dec, scheduler,
                            val_timesteps, fixed_optical_emb,
                        )
                        tensor_to_pil(generated[0]).save(os.path.join(sample_dir, os.path.basename(image_path)))

                    metrics = evaluate_image_folders(
                        os.path.join(config.dataset_folder, "testA"), sample_dir, os.path.join(config.dataset_folder, "testB"), 
                        device=device, report_path=os.path.join(os.path.dirname(sample_dir), "metrics.txt"),
                    )
                    logs.update({f"val/{key}": value for key, value in metrics.items()})
                    logger.info("第 %d 步验证: %s", global_step, metrics)
                    gc.collect()
                    torch.cuda.empty_cache()

            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

        if global_step >= config.max_train_steps:
            break

    progress_bar.close()
    accelerator.end_training()
    logger.info("训练结束，共 %d 步", global_step)


def main():
    train(resolve_train_config())


if __name__ == "__main__":
    main()
