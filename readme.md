# Efficient Contrastive Finetuning of Latent Diffusion Models for One\-Step Unpaired SAR\-to\-Optical Image Translation

**Official implementation for CFDM.**

The source code will be released after the paper is published\.

---

## 📌 Abstract / 摘要

SAR\-to\-optical image translation \(S2OIT\) is vital for enhancing SAR image interpretability but faces challenges in balancing generation fidelity with inference efficiency under unpaired settings\. This letter proposes Contrastive Finetuning Diffusion Models \(CFDM\), a novel one\-step unpaired translation framework\. Specifically, we adapt a pretrained latent diffusion model using LoRA and residual\-connected layers to enable high\-fidelity, end\-to\-end generation with minimal computational cost\. To preserve intrinsic geometries, a feature\-space contrastive learning strategy is introduced, leveraging a DINO\-based extractor to enforce patch\-level semantic consistency\. Furthermore, a maximum mean discrepancy \(MMD\) regularization is incorporated to explicitly align global statistical distributions between domains\. Extensive experiments on SAR2Opt and SEN1\-2 datasets demonstrate that CFDM outperforms state\-of\-the\-art methods in both perceptual quality and structural preservation, providing a novel and robust methodological paradigm for addressing the inherent domain gap in S2OIT tasks\.

**Keywords：** Contrastive learning, diffusion models, maximum mean discrepancy, parameter efficient finetuning, SAR\-to\-optical image translation\.

---

## 🖼️ Method Overview / 方法概述

- 本文面向SAR到光学图像转换（S2OIT）任务，基于预训练扩散模型构建参数高效微调方法，融合对抗学习、域自适应与对比学习联合优化生成器，兼顾图像像素保真、分布对齐与结构一致性。

- 完整训练框架包含三大核心网络，分别为生成器、鉴别器和特征提取器。

- 推理阶段只保留生成器模型即可实现单步图像风格翻译。

---

## 💻 Environment Requirements / 环境依赖

本项目代码基于以下环境开发与测试：

- python == 3\.10

- pyTorch == 2\.4\.0

- torchvision == 0\.19

- diffusers == 0\.25\.1

- accelerate == 1\.12\.0

- peft == 0\.10

- vision\_aided\_loss == 0\.1\.0

- cuda == 12\.1

---

## 📊 Dataset Preparation / 数据集准备

本论文实验基于公开数据集 **sar2opt** 与 **sen12** 进行筛选与测试。需手动配置数据集根目录，标准文件夹结构如下：

```Plain Text
dataset_root/
├── trainA/   # 训练集源域图像         
├── trainB/   # 训练集目标域图像     
├── testA/    # 测试集源域图像      
└── testB/    # 测试集目标域图像       
```

## Citing

```Plain Text
@article{yuweb2026cfdm,
  title={Efficient Contrastive Finetuning of Latent Diffusion Models for One-Step Unpaired SAR-to-Optical Image Translation},
  author={Wenbo Yu, Tian Tian, Jiamu Li, and Feng Zhou},
}
```
