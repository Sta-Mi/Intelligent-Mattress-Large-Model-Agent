# Intelligent-Mattress-Large-Model-Agent

面向智能床垫压力分布数据的大模型智能体项目。当前仓库已经包含两条可直接推进的公开基线：

- **BodyMAP（CVPR 2024）姿态/人体网格基线**：位于 `BodyPressure/BodyMAP/`，用于从深度图与压力图联合预测 3D 人体网格、3D 姿态/形状和人体表面压力分布。
- **ConvNeXt V2 Base 身份识别基线**：位于 `BodyPressure/identity_recognition/`，用于在 SLP cleaned pressure/depth `.npy` 数据上做 closed-set subject ID 分类。

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
