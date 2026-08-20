# 不依赖 SLP 的细粒度姿态基线

## 选择

本目录实现 **PressurePoseTransformer**：仅输入 `64×27` 压力图，使用 ViT-style
taxel token encoder 和 24 个可学习 joint query，经 cross-attention 输出压力垫坐标系中的
24 个 3D 身体关节（米）。它是一个可复现的现代强基线，不冒充已有论文的 SOTA 数字；
真正的先进性需要在固定公开测试集上用 MPJPE/PCK 与其他方法实测。

选择它而不是仓库中的 BodyMAP，是因为这里完全不读取 SLP、SLP-3Dfits、RGB 或深度图，
也不需要 SMPL 模型文件。唯一数据依赖是公开、无压缩包密码的 **BodyPressureSD** 压力
pickle；其中已经包含压力图和 24 个 SMPL joint 的米制标注。因而它能回答“身体部位在哪里”，
而不只是输出躺姿类别。

## 数据与复现实验

从 Harvard Dataverse 的 BodyPressureSD 下载页面下载 `synth/*.p` 文件，或直接运行
`bash BodyPressure/pose_estimation/download_pressure_only.sh /path/to/BodyPressureSD`。只训练本
基线不需要 `synth_depth`，也不需要 `GT_BP_data`。目录结构为：

```text
DATA_ROOT/synth/train_slp_lay_f_1to40_8549.p
...
```

默认列表采用 body-shape range `1--70` 训练、`71--80` 验证，且三类姿态和两个性别均出现；
不要随机按帧切分，否则同一合成人体形状会泄漏到验证集。运行：

```bash
python -m BodyPressure.pose_estimation.train --data_root /path/to/BodyPressureSD \
  --train_files BodyPressure/pose_estimation/train_files.txt \
  --val_files BodyPressure/pose_estimation/val_files.txt \
  --device cuda --epochs 100 --batch_size 64
```

`metrics.jsonl` 记录 train/validation MPJPE（毫米），`best_model.pt` 按最低 validation MPJPE
保存。合成验证结果只能用于架构消融；要声称真实床垫精度，必须用自行采集且与训练 subject
隔离的真实压力图 + 关节标注测试，不能用不可获得的 SLP 结果替代。

## 可替代数据的边界

1. **BodyPressureSD（首选预训练）**：公开且有 24 个 3D joints，最适合先打通细粒度定位；
   缺点是合成域，必须报告真实域差距。
2. **PmatData/PhysioNet**：公开压力数据适合姿态类别、呼吸或占床分析，但不应在未核对具体
   版本标注时宣称可直接监督 24 点 3D 姿态。
3. **自采少量真实集（首选微调/测试）**：使用顶视 RGB-D 或 mocap 仅在采集阶段生成伪标签，
   部署模型仍只读压力；按受试者划分并人工审核肩、肘、腕、髋、膝、踝。

如果目标是 SMPL mesh 而非关节定位，可在本模型的 joint tokens 后增加 SMPL 参数 head，
但那会重新引入受许可约束的 SMPL 文件；在没有真实 mesh 标签时不建议把它作为第一步。
