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

`save_inference.py` 完成时只表示预测文件已生成；`metrics.py` 的进度条完成也不等于已经看到数值结果。新版 `metrics.py` 会将 uncover/cover1/cover2/overall 的 MPJPE、PVE、人体尺寸误差、v2vP/1EA/2EA 和分身体部位 pressure error 同时打印并写入 `<save_path>/metrics.json`，TensorBoard 事件仍保留在 `<save_path>/metric_overall/`。预训练权重究竟是 `both` 还是 `depth` 模态，应以对应 `exp.json` 的 `modality` 字段为准；即使是 depth-only，loader 仍会打印并传递 pressure tensor，模型内部配置才决定是否使用它。

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
  --batch_size 32
```

`real_all.txt` 包含 101 个 subject，因此随机 top-1 约为 `1/101=0.0099`，随机 top-5 约为 `5/101=0.0495`。仅第 1 个 epoch 出现 `train_acc≈0.007`、`val_acc≈0.010`、`val_top5≈0.050` 可能只是随机初始化；但若 5--10 个 epoch 后仍完全停留在这些数值，则**不正常**，说明模型没有学习。训练日志会同时打印 train/val loss：正常训练时 train loss 应从随机分类的 `ln(101)≈4.615` 明显下降。ConvNeXt V2 输入会按其 ImageNet 预训练配置归一化；训练脚本默认给随机初始化分类头使用 `--head_lr_mult 10.0`，即 head 学习率为 backbone 学习率的 10 倍。

若完整训练停在随机水平，先用同一小批样本做过拟合检查；下面的 2-subject/4-pose 训练集与验证集故意相同，准确率应很快明显高于随机值 `0.5`：

```bash
python BodyPressure/identity_recognition/train_identity.py \
  --device cuda --model small_cnn --mode pressure --epochs 20 \
  --split_strategy range \
  --limit_subjects 2 --train_pose_start 0 --train_pose_end 4 \
  --val_pose_start 0 --val_pose_end 4 \
  --out_dir /tmp/identity_overfit_check
```

训练 DataLoader 不再丢弃最后一个不足 `batch_size` 的 batch。旧版本在上述 8 个样本、默认 `batch_size=32` 时因 `drop_last=True` 实际产生 **0 个训练 batch**，所以会显示 `train_loss=0.0000`、`train_acc=0.0000`；这不是数据或模型结果。新版启动日志会打印 train/val batch 数，并在实际处理 0 个训练样本时直接报错。

该检查的通过标准是 train loss 持续下降并最终达到 `train_acc=1.0`；因为 train/val 故意使用完全相同的 8 个样本，`val_acc=1.0` 仅证明数据读取、标签、反向传播和模型保存链路正常，**不能作为泛化结果**。通过后应先运行压力原生 SmallCNN 的正式 pose-holdout 基线，再与 ConvNeXt V2 比较：

```bash
python BodyPressure/identity_recognition/train_identity.py \
  --device cuda --model small_cnn --mode pressure \
  --epochs 100 --batch_size 32 --lr 1e-3 --head_lr_mult 1 \
  --split_strategy range \
  --train_pose_start 0 --train_pose_end 35 \
  --val_pose_start 35 --val_pose_end 45 \
  --out_dir /home/shnh/DATA/zjy/BodyMAP_identity_smallcnn_pose_holdout
```

这里令 `--head_lr_mult 1`，是因为 SmallCNN 的特征提取器和分类头都是随机初始化的，不需要对分类头使用预训练模型专用的差分学习率。如果 SmallCNN 的 pose-holdout 指标明显高于随机值而 ConvNeXt 仍为随机水平，问题应定位为 RGB→压力的迁移/预训练域差异，而不是 Dataset 或标签错误。

SLP 的 45 个姿态不是同分布随机帧：BodyMAP 将 `[0,15)` 作为 `lay`、`[15,45)` 作为 `side`。因此 `[0,35)` 训练、`[35,45)` 验证是偏向 side 姿态的严格跨姿态测试。若训练准确率持续升高而验证 top-1 仅约 `1%--3%`、验证 loss 持续升高，结论是严重的姿态过拟合，而不是“模型没有训练”。此时不应继续增加 epoch；应报告最佳验证 epoch，并进一步比较分层姿态划分、度量学习和压力域自监督预训练。

训练脚本现在默认使用 `--split_strategy stratified --pose_folds 5 --pose_fold 4`：从 45 个姿态中每 5 个留出 1 个，验证姿态为 `[4,9,14,19,24,29,34,39,44]`，其余 36 个用于训练。这样 lay 和 side 都同时出现在训练/验证中，适合作为主 closed-set 基线；原来的连续区间应通过 `--split_strategy range` 显式启用，并作为更严格的 cross-posture 补充实验。

SmallCNN 使用 `AdaptiveAvgPool2d(1)`，会把压力特征压成全局均值，适合作为管线检查，但不适合作为最终压力身份模型。若它在分层划分上出现 train accuracy `>85%` 而 val top-1 只有 `2%--4%`，应停止继续训练。下一步使用保留粗粒度身体接触位置的 PressureCNN：

```bash
python BodyPressure/identity_recognition/train_identity.py \
  --device cuda --model pressure_cnn --mode pressure \
  --epochs 60 --batch_size 32 --lr 3e-4 --head_lr_mult 1 \
  --weight_decay 1e-3 --split_strategy stratified \
  --pose_folds 5 --pose_fold 4 \
  --out_dir /home/shnh/DATA/zjy/BodyMAP_identity_pressurecnn_stratified
```

PressureCNN 在该分层协议上若达到约 `22%` sample top-1、`45%` top-5 和 `66%` subject-level accuracy，已经显著高于 `0.99%/4.95%/0.99%` 的随机基线，可作为有效的压力原生基线。训练准确率继续上升而验证 loss 在约第 10--20 轮后回升表示开始过拟合；脚本默认在 subject accuracy 连续 15 轮没有改善时提前停止，并在 checkpoint 中记录最佳 epoch、验证指标和完整配置。

五个 pose fold 的最佳 subject accuracy 为 `75.25%、77.23%、78.22%、75.25%、66.34%` 时，均值为 `74.46%`、样本标准差约 `4.72%`，说明多姿态聚合身份信号较稳定，可以进入 ArcFace/多帧 embedding 阶段。注意：对全部 45 个姿态评估会混入 36 个训练姿态；例如 `acc_sample≈75%`、`acc_subject=100%` 不能作为独立测试结果。评估脚本默认从新版 checkpoint 读取精确的 `val_pose_indices`；只有显式传入 `--all_poses` 才允许包含训练姿态的诊断评估。

ArcFace + embedding 训练（先运行一个 fold）：

```bash
python BodyPressure/identity_recognition/train_identity.py \
  --device cuda --model pressure_arcface --mode pressure \
  --embedding_dim 256 --arcface_scale 30 --arcface_margin 0.3 \
  --samples_per_subject 4 --supcon_weight 0.05 --supcon_temperature 0.07 \
  --epochs 80 --batch_size 32 --lr 3e-4 --head_lr_mult 1 \
  --weight_decay 1e-3 --split_strategy stratified \
  --pose_folds 5 --pose_fold 4 --early_stopping_patience 15 \
  --out_dir /home/shnh/DATA/zjy/BodyMAP_identity_arcface_fold4
```

`pressure_arcface` 使用保留 `8×4` 空间网格的压力编码器，将每帧映射为 256 维 L2-normalized embedding。训练时只对真实类别施加 ArcFace angular margin；验证时不施加 margin，而使用归一化类别权重的 cosine logits，避免人为压低验证指标。checkpoint 同时保存 encoder 与 ArcFace head，评估后还会生成 `embeddings.pt`，供余弦相似度验证和多帧模板聚合使用。

ArcFace fold 4 若达到约 `22.9%` sample top-1、`65.35%` subject accuracy，而普通 PressureCNN 为约 `22.1%/66.34%`，表示 ArcFace 暂未改善 closed-set 分类，但不能只据此否定 embedding。先在严格 held-out pose embedding 上计算 verification ROC-AUC、EER 和 TAR@FAR：

```bash
python BodyPressure/identity_recognition/eval_identity.py \
  --checkpoint /home/shnh/DATA/zjy/BodyMAP_identity_arcface_fold4/best_model.pt \
  --split real_all.txt --mode pressure --device cuda \
  --out_dir /home/shnh/DATA/zjy/BodyMAP_identity_arcface_fold4/heldout_eval

python BodyPressure/identity_recognition/eval_verification.py \
  --embeddings /home/shnh/DATA/zjy/BodyMAP_identity_arcface_fold4/heldout_eval/embeddings.pt
```

`eval_verification.py` 对 held-out embeddings 的所有上三角样本对计算 cosine similarity；同一 subject 为正对，不同 subject 为负对，并输出 ROC-AUC、EER、EER threshold 以及 TAR@FAR=0.1/0.01/0.001。不要使用 `--all_poses` 生成的 embedding 计算正式 verification 指标。

若仅用随机 batch 训练 ArcFace，101 个身份下同一 batch 通常缺少同身份正样本，可能出现 closed-set subject accuracy 尚可但 verification 很差（例如 ROC-AUC 约 `0.60`、EER 约 `0.43`）。ArcFace 训练现在默认使用 P×K sampler：`batch_size=32, K=4` 表示每批 8 个 subject、每人 4 个姿态；同时加入低权重 `0.05` 的 supervised contrastive loss，直接拉近同身份跨姿态 embedding、推远批内其他身份。旧 checkpoint 不受影响，但需重新训练才能获得该改进。

如果 P×K + SupCon 后 held-out ROC-AUC 仍约 `0.58`、EER 约 `0.44`，说明按 subject classification accuracy 选 checkpoint 与 embedding verification 目标不一致。ArcFace 训练现在每个 epoch 直接在 909 个 held-out embeddings 上计算 ROC-AUC/EER/TAR@FAR，并按“最高 ROC-AUC、再最低 EER”保存和 early-stop；普通 softmax 模型仍按 subject/sample accuracy 选择。建议重新训练时先将 `--supcon_weight` 降为 `0.05`，避免强 SupCon 在姿态差异远大于身份差异时过度拉扯 embedding。

若 verification-aware 最佳 checkpoint 仍只有 ROC-AUC 约 `0.59`、EER 约 `0.43`，单帧对单帧验证路线已经达到当前表示的瓶颈。智能床垫实际注册应使用多姿态 enrollment template：用 checkpoint 中 36 个训练姿态的 embedding 为每人求均值并 L2 normalize，只用 9 个 held-out 姿态作为 probe。该协议允许 enrollment 使用注册数据，但绝不把 enrollment 帧当 probe：

```bash
python BodyPressure/identity_recognition/eval_template_identity.py \
  --checkpoint /home/shnh/DATA/zjy/BodyMAP_identity_arcface_auc_fold4/best_model.pt \
  --split real_all.txt --mode pressure --device cuda \
  --out_dir /home/shnh/DATA/zjy/BodyMAP_identity_arcface_auc_fold4/template_eval
```

输出的 `template_metrics.json` 包含 probe-to-template 的 top-1/top-5、subject accuracy、ROC-AUC、EER 和 TAR@FAR；`templates.pt` 是 101 个用户的注册模板。这个结果比任意两张 held-out 单帧互相比对更贴近连续监测床垫的实际身份注册/识别方式。

fold 4 的多姿态模板达到 ROC-AUC `0.7726`、EER `0.2913`、TAR@FAR=1% `0.2112`，相较单帧 pair 的约 `0.59/0.43/0.03` 有实质提升，说明 enrollment template 路线有效；但 `acc_subject=58.42%` 仍低于 PressureCNN softmax 多帧聚合基线，因此应保留两条指标线，不把 verification 提升表述成所有身份指标都提升。下一步对 5 个 fold 重复 template evaluation，并汇总均值、样本标准差、最小值和最大值：

```bash
python BodyPressure/identity_recognition/summarize_folds.py \
  /home/shnh/DATA/zjy/BodyMAP_identity_arcface_auc_fold{0,1,2,3,4}/template_eval/template_metrics.json \
  --out /home/shnh/DATA/zjy/BodyMAP_identity_arcface_template_5fold.json
```

只有五折 template ROC-AUC/EER 都稳定后，再实现姿态条件模板或在线模板更新；单个 fold 不能作为最终结论。

当前 ConvNeXt V2 由 `timm` 创建，且可直接加载已下载的本地 checkpoint，**不需要再克隆 ConvNeXt-V2 仓库**。只有准备复现官方 FCMAE 预训练流程时才需要官方仓库。DINOv2/InsightFace 同样不应仅为运行现有 softmax 基线而克隆：实现压力域 DINO 自监督时再克隆 DINOv2；ArcFace loss 可以作为本项目中的小型 PyTorch 模块实现，无需引入整个 InsightFace 人脸识别工程。

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
  --mode pressure
```

## 项目路线图

中文实施路线图见 `docs/implementation-roadmap-zh.md`。

当前完成度、身份五折结果和下一阶段执行顺序见 [`docs/project-status-zh.md`](docs/project-status-zh.md)。
