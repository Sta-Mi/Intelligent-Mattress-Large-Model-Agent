# Intelligent-Mattress-Large-Model-Agent

面向智能床垫压力分布数据的大模型智能体项目。当前仓库已经包含两条可直接推进的公开基线：

- **BodyMAP（CVPR 2024）姿态/人体网格基线**：位于 `BodyPressure/BodyMAP/`，用于从深度图与压力图联合预测 3D 人体网格、3D 姿态/形状和人体表面压力分布。
- **ConvNeXt V2 Base 身份识别基线**：位于 `BodyPressure/identity_recognition/`，用于在 SLP cleaned pressure/depth `.npy` 数据上做 closed-set subject ID 分类。
- **身份识别前沿模型调研**：见 [`BodyPressure/identity_recognition/BASELINES.md`](BodyPressure/identity_recognition/BASELINES.md)，包含 DINOv2、MAE、ArcFace、ConvNeXt V2、Swin V2 的源码和建议实验协议。

## 已下载基线的建议用法

### 1. BodyMAP 细粒度姿态估计

BodyMAP 是当前仓库中最适合作为“压力图 + 深度图到 3D 人体网格/身体部位压力”的前沿基线。优先使用官方配置：

```bash
cd BodyPressure/BodyMAP/PMM
python main.py ../model_config/Conv.json
```

训练完成后，按 BodyMAP README 的流程保存推理结果并计算 3D pose、3D shape、3D pressure metrics。

### 2. ConvNeXt V2 Base 身份识别

身份识别脚本默认使用 `convnextv2_base`。若本地已经下载以下任一 checkpoint，代码会优先从本地加载；否则会回退到 `timm` 的在线 pretrained 权重：

- `convnextv2_base.fcmae_ft_in22k_in1k.safetensors`
- `convnextv2_base_22k_224_ema.pt`

默认本地权重目录为：

```text
/home/shnh/DATA/zjy/BodyMAP_identity_pretrained
```

也可以通过环境变量覆盖：

```bash
export BODYMAP_IDENTITY_PRETRAINED_DIR=/path/to/BodyMAP_identity_pretrained
```

快速 smoke test：

```bash
python BodyPressure/identity_recognition/train_identity.py \
  --device cpu \
  --model small_cnn \
  --epochs 1 \
  --limit_subjects 2 \
  --train_pose_end 2 \
  --val_pose_start 2 \
  --val_pose_end 4
```

正式训练 ConvNeXt V2 Base：

```bash
python BodyPressure/identity_recognition/train_identity.py \
  --device cuda \
  --model convnextv2_base \
  --mode pressure \
  --epochs 30 \
  --batch_size 32 \
  --train_pose_end 35 \
  --val_pose_start 35
```

`real_all.txt` 包含 101 个 subject，因此随机 top-1 约为 `1/101=0.0099`，随机 top-5 约为 `5/101=0.0495`。仅第 1 个 epoch 出现 `train_acc≈0.007`、`val_acc≈0.010`、`val_top5≈0.050` 可能只是随机初始化；但若 5--10 个 epoch 后仍完全停留在这些数值，则**不正常**，说明模型没有学习。训练日志会同时打印 train/val loss：正常训练时 train loss 应从随机分类的 `ln(101)≈4.615` 明显下降。ConvNeXt V2 输入会按其 ImageNet 预训练配置归一化；训练脚本默认给随机初始化分类头使用 `--head_lr_mult 10.0`，即 head 学习率为 backbone 学习率的 10 倍。

若完整训练停在随机水平，先用同一小批样本做过拟合检查；下面的 2-subject/4-pose 训练集与验证集故意相同，准确率应很快明显高于随机值 `0.5`：

```bash
python BodyPressure/identity_recognition/train_identity.py \
  --device cuda --model small_cnn --mode pressure --epochs 20 \
  --limit_subjects 2 --train_pose_start 0 --train_pose_end 4 \
  --val_pose_start 0 --val_pose_end 4 \
  --out_dir /tmp/identity_overfit_check
```

训练 DataLoader 不再丢弃最后一个不足 `batch_size` 的 batch。旧版本在上述 8 个样本、默认 `batch_size=32` 时因 `drop_last=True` 实际产生 **0 个训练 batch**，所以会显示 `train_loss=0.0000`、`train_acc=0.0000`；这不是数据或模型结果。新版启动日志会打印 train/val batch 数，并在实际处理 0 个训练样本时直接报错。

检查实际压力数组中负值、正值和超过裁剪上限的比例：

```bash
python BodyPressure/identity_recognition/inspect_data.py \
  /home/shnh/DATA/zjy/slp_real_cleaned/pressure_recon_Pplus_gt_0to102.npy
```

`pressure_recon_Pplus` 是重建结果，出现负值并不代表真实存在“负压力”；当前身份 Dataset 会把负值裁剪为 0。若 `positive_ratio` 只有几个百分点，输入会非常稀疏。模型现在保持原始 `64×27` 纵横比缩放并补边到 `224×224`，不再把人体压力图横向拉伸成正方形。

> 注意：身份识别是 closed-set 分类时，训练集和验证集必须包含同一批 subject，只按姿态/时间/session 留出验证样本。`real_train.txt` 与 `real_val.txt` 是 BodyMAP 姿态估计使用的 subject-disjoint 划分，不适合作为 closed-set 身份识别的默认 train/val 组合。

评估：

```bash
python BodyPressure/identity_recognition/eval_identity.py \
  --checkpoint /home/shnh/DATA/zjy/BodyMAP_identity/best_model.pt \
  --split real_all.txt \
  --mode pressure \
  --pose_start 35
```

## 项目路线图

中文实施路线图见 `docs/implementation-roadmap-zh.md`。
