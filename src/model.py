"""
模型构建：单步扩散调度器、VAE 前向改写、LoRA 注入与翻译模型封装。
"""

import torch
import torch.nn as nn
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from peft import LoraConfig
from transformers import AutoTokenizer, CLIPTextModel

# VAE 解码器 skip 卷积的 (输入通道, 输出通道)
SKIP_CONV_CHANNELS = ((512, 512), (256, 512), (128, 512), (128, 256))

# VAE 中注入 LoRA 的层名
VAE_LORA_TARGETS = [
    "conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
    "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4",
    "to_k", "to_q", "to_v", "to_out.0",
]


def make_1step_sched(pretrained_name="stabilityai/sd-turbo", device="cuda"):
    """构造 1 步扩散采样器（推理时只用 t=999 一步去噪）。"""
    scheduler = DDPMScheduler.from_pretrained(pretrained_name, subfolder="scheduler")
    scheduler.set_timesteps(1, device=device)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
    return scheduler


def vae_encoder_forward(self, sample):
    """替换 VAE 编码器前向：缓存各下采样层输出，供解码器 skip 连接使用。"""
    sample = self.conv_in(sample)                       # (B, 3, H, W) -> (B, 128, H, W)
    down_blocks = []
    for down_block in self.down_blocks:
        down_blocks.append(sample)
        sample = down_block(sample)
    sample = self.mid_block(sample)
    sample = self.conv_act(self.conv_norm_out(sample))
    sample = self.conv_out(sample)                      # -> (B, 8, H/8, W/8)
    self.current_down_blocks = down_blocks
    return sample


def vae_decoder_forward(self, sample, latent_embeds=None):
    """替换 VAE 解码器前向：把编码器缓存的 4 级特征逐级加回解码路径。"""
    sample = self.conv_in(sample)
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
    sample = self.mid_block(sample, latent_embeds).to(upscale_dtype)

    if self.ignore_skip:
        for up_block in self.up_blocks:
            sample = up_block(sample, latent_embeds)
    else:
        skip_convs = [self.skip_conv_1, self.skip_conv_2, self.skip_conv_3, self.skip_conv_4]
        for index, up_block in enumerate(self.up_blocks):
            # 从最深层特征开始取，逐级上采样恢复到 (B, 128, H, W)
            skip = skip_convs[index](self.incoming_skip_acts[::-1][index] * self.gamma)
            sample = up_block(sample + skip, latent_embeds)

    sample = self.conv_norm_out(sample) if latent_embeds is None else self.conv_norm_out(sample, latent_embeds)
    return self.conv_out(self.conv_act(sample))


def attach_skip_connections(vae, device):
    """给 VAE 挂上 4 个可学习 1x1 skip 卷积，并改写编码/解码前向。

    skip 卷积初始化为 1e-5，实现ZeroConv效果。
    """
    vae.encoder.forward = vae_encoder_forward.__get__(vae.encoder, vae.encoder.__class__)
    vae.decoder.forward = vae_decoder_forward.__get__(vae.decoder, vae.decoder.__class__)
    for index, (in_channels, out_channels) in enumerate(SKIP_CONV_CHANNELS, start=1):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        nn.init.constant_(conv.weight, 1e-5)
        setattr(vae.decoder, f"skip_conv_{index}", conv.to(device).requires_grad_(True))
    vae.decoder.ignore_skip = False
    vae.decoder.gamma = 1
    return vae


def initialize_unet(rank, pretrained_name="stabilityai/sd-turbo"):
    """加载 UNet 并按编码器 / 解码器 / 其它三组注入 LoRA。

    Returns:
        (unet, (编码器目标模块名, 解码器目标模块名, 其它目标模块名))，
        模块名用于保存与加载权重。
    """
    unet = UNet2DConditionModel.from_pretrained(pretrained_name, subfolder="unet")
    unet.requires_grad_(False)
    unet.train()

    patterns = (
        "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_in",
        "conv_shortcut", "conv_out", "proj_out", "proj_in", "ff.net.2", "ff.net.0.proj",
    )
    encoder, decoder, others = [], [], []
    for name, _ in unet.named_parameters():
        if "bias" in name or "norm" in name:
            continue
        for pattern in patterns:
            if pattern not in name:
                continue
            if "down_blocks" in name or "conv_in" in name:
                encoder.append(name.replace(".weight", ""))
            elif "up_blocks" in name:
                decoder.append(name.replace(".weight", ""))
            else:
                others.append(name.replace(".weight", ""))
            break

    for adapter_name, target_modules in (
        ("default_encoder", encoder),
        ("default_decoder", decoder),
        ("default_others", others),
    ):
        unet.add_adapter(
            LoraConfig(r=rank, lora_alpha=rank, init_lora_weights="gaussian", target_modules=target_modules),
            adapter_name=adapter_name,
        )
    unet.set_adapters(["default_encoder", "default_decoder", "default_others"])
    return unet, (encoder, decoder, others)


def initialize_vae(rank, pretrained_name="stabilityai/sd-turbo", device="cuda"):
    """加载 VAE，添加 skip 卷积与 LoRA。

    Returns:
        (vae, VAE LoRA 目标模块名)
    """
    vae = AutoencoderKL.from_pretrained(pretrained_name, subfolder="vae")
    vae.requires_grad_(False)
    attach_skip_connections(vae, device)
    vae.requires_grad_(True)
    vae.train()
    vae.add_adapter(
        LoraConfig(r=rank, init_lora_weights="gaussian", target_modules=list(VAE_LORA_TARGETS)),
        adapter_name="vae_skip",
    )
    return vae, list(VAE_LORA_TARGETS)


class VAE_encode(nn.Module):
    """VAE 编码包装：图像 (B, 3, H, W) -> latent (B, 4, H/8, W/8)。"""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, images):
        return self.vae.encode(images).latent_dist.sample() * self.vae.config.scaling_factor


class VAE_decode(nn.Module):
    """VAE 解码包装：latent (B, 4, H/8, W/8) -> 图像 (B, 3, H, W)。"""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latents):
        assert self.vae.encoder.current_down_blocks is not None, "解码前必须先执行一次 VAE_encode"
        self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
        images = self.vae.decode(latents / self.vae.config.scaling_factor).sample
        return images.clamp(-1, 1)


class CFDM(nn.Module):
    """单步扩散图像翻译模型：SD 主干 + UNet/VAE LoRA。

    训练与推理共用；推理时通过 load_checkpoint 加载微调权重并把 LoRA 融合进主干。
    """

    def __init__(self, pretrained_path=None, pretrained_name="stabilityai/sd-turbo", device="cuda", timestep=999):
        super().__init__()
        self.device = torch.device(device)
        self.timestep = timestep

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name, subfolder="tokenizer", use_fast=False)
        self.text_encoder = CLIPTextModel.from_pretrained(pretrained_name, subfolder="text_encoder")
        self.scheduler = make_1step_sched(pretrained_name, self.device)
        self.unet = UNet2DConditionModel.from_pretrained(pretrained_name, subfolder="unet")
        self.vae = AutoencoderKL.from_pretrained(pretrained_name, subfolder="vae")
        attach_skip_connections(self.vae, self.device)

        self.text_encoder.requires_grad_(False)
        self.vae_enc = VAE_encode(self.vae)
        self.vae_dec = VAE_decode(self.vae)
        for module in (self.text_encoder, self.unet, self.vae_enc, self.vae_dec):
            module.to(self.device)

        if pretrained_path is not None:
            self.load_checkpoint(pretrained_path)

    def load_checkpoint(self, pretrained_path):
        """加载微调权重（.pkl）并融合 LoRA，字段格式见 train.py 中的保存逻辑。"""
        state = torch.load(pretrained_path, map_location="cpu")
        rank_unet = state["rank_unet"]

        for adapter_name, module_names, key in (
            ("default_encoder", state["l_target_modules_encoder"], "sd_encoder"),
            ("default_decoder", state["l_target_modules_decoder"], "sd_decoder"),
            ("default_others", state["l_modules_others"], "sd_other"),
        ):
            self.unet.add_adapter(
                LoraConfig(r=rank_unet, lora_alpha=rank_unet, init_lora_weights="gaussian",
                           target_modules=module_names),
                adapter_name=adapter_name,
            )
            for name, param in self.unet.named_parameters():
                if "lora" in name and adapter_name in name:
                    param.data.copy_(state[key][name.replace(f".{adapter_name}.weight", ".weight")])
        self.unet.set_adapters(["default_encoder", "default_decoder", "default_others"])
        self.unet.fuse_lora()
        self.unet.enable_xformers_memory_efficient_attention()

        self.vae.add_adapter(
            LoraConfig(r=state["rank_vae"], init_lora_weights="gaussian",
                       target_modules=state["vae_lora_target_modules"]),
            adapter_name="vae_skip",
        )
        self.vae.decoder.gamma = 1
        # sd_vae_enc 是带 vae_skip 适配器的完整 VAE 权重（含 skip 卷积与 LoRA），整体加载即可
        self.vae_enc.load_state_dict(state["sd_vae_enc"])
        return self

    @staticmethod
    def forward_with_networks(images, vae_enc, unet, vae_dec, scheduler, timesteps, text_emb):
        """给定网络组件执行一次单步翻译。

        Args:
            images: 输入图像 (B, 3, H, W)，取值 [-1, 1]。
            vae_enc / unet / vae_dec: 编码器、UNet、解码器。
            scheduler: 单步调度器。
            timesteps: (B,) 时间步张量。
            text_emb: 目标域文本嵌入 (B, 77, 1024)。

        Returns:
            生成图像 (B, 3, H, W)，取值 [-1, 1]。
        """
        latents = vae_enc(images).to(images.dtype)                      # (B, 3, H, W) -> (B, 4, H/8, W/8)
        noise_pred = unet(latents, timesteps, encoder_hidden_states=text_emb).sample
        denoised = torch.stack([
            scheduler.step(noise_pred[i], timesteps[i], latents[i], return_dict=True).prev_sample
            for i in range(images.shape[0])
        ])                                                              # 单步去噪后的 latent
        return vae_dec(denoised)                                        # -> (B, 3, H, W)

    @staticmethod
    def get_trainable_params(unet, vae):
        """收集生成器可训练参数：UNet conv_in、三组 LoRA、VAE LoRA 与 skip 卷积。"""
        unet.conv_in.requires_grad_(True)
        unet.set_adapters(["default_encoder", "default_decoder", "default_others"])
        params = list(unet.conv_in.parameters())
        params += [p for name, p in unet.named_parameters() if "lora" in name and "default" in name]
        params += [p for name, p in vae.named_parameters() if "lora" in name and "vae_skip" in name]
        for index in range(1, len(SKIP_CONV_CHANNELS) + 1):
            params += list(getattr(vae.decoder, f"skip_conv_{index}").parameters())
        return params

    @torch.no_grad()
    def translate(self, images, caption_emb):
        """对输入图像执行单步彩色化。

        Args:
            images: 输入图像 (B, 3, H, W)，取值 [-1, 1]，已在模型设备上。
            caption_emb: 目标域（光学）文本嵌入 (1, 77, 1024)。

        Returns:
            彩色化结果 (B, 3, H, W)，取值 [-1, 1]。
        """
        caption_emb = caption_emb.to(images.device, dtype=images.dtype)
        if caption_emb.shape[0] != images.shape[0]:
            caption_emb = caption_emb.repeat(images.shape[0], 1, 1)
        timesteps = torch.full((images.shape[0],), self.timestep, device=images.device, dtype=torch.long)
        return self.forward_with_networks(
            images, self.vae_enc, self.unet, self.vae_dec, self.scheduler, timesteps, caption_emb
        )
