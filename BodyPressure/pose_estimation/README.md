# 不依赖 SLP 的细粒度姿态基线

## 选择

本目录实现 **PressurePoseTransformer**：仅输入 `64×27` 压力图，使用 ViT-style
taxel token encoder 和 24 个可学习 joint query，经 cross-attention 输出 BodyMAP 合成床坐标系中的
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

你当前给出的路径可以直接作为参数（也可以直接传到最后一级 `synth`）：

```bash
python -m BodyPressure.pose_estimation.train \
  --data_root /home/zjy/Intelligent_Mattress_Large_Model_Agent/Intelligent-Mattress-Large-Model-Agent/BodyPressure/data_BP/synth \
  --train_files BodyPressure/pose_estimation/train_files.txt \
  --val_files BodyPressure/pose_estimation/val_files.txt --device cuda
```

loader 会采用与 BodyMAP 官方读取器相同的 `body_volume + gender` 质量换算，将模拟器压力
强度转换为 mmHg 后裁剪/归一化；不能直接把 pickle 中的原始数值当作 mmHg。模型会把
`64×27` 自动补边到 patch size 的整数倍，最右侧 3 列不会再被卷积静默丢弃。
训练 sampler 会在文件内随机样本并随机文件顺序，但保持每个 batch 的 pickle 局部性；这是
因为 pickle 不支持样本级随机读取，全局随机索引会反复重载数百 MB 文件，训练会异常缓慢。

你已有的 `synth_depth` 与这些压力文件逐样本对应，可用于后续 **pressure-only 与
pressure+depth** 的受控消融；但本基线有意不读取它。这样得到的结果才能回答“只靠智能床垫
压力阵列能定位到什么程度”，也不会把顶视深度信息带来的提升误算成压力模型能力。

`metrics.jsonl` 记录 train/validation MPJPE（毫米），`best_model.pt` 按最低 validation MPJPE
保存。合成验证结果只能用于架构消融；要声称真实床垫精度，必须用自行采集且与训练 subject
隔离的真实压力图 + 关节标注测试，不能用不可获得的 SLP 结果替代。

你这次 100 epoch 日志中的最低 validation MPJPE 是 **epoch 87 的 77.93 mm**，不是最后
epoch 的 81.75 mm；训练仍持续下降、验证在约 79--83 mm 波动，属于轻度泛化平台期。新版
训练默认使用 cosine learning-rate decay，并在连续 20 epoch 无改善时提前停止。PyTorch 输出的
`enable_nested_tensor ... norm_first=True` 是性能提示，不影响数值正确性。

不要只汇报一个 overall MPJPE。用已经保存的最佳 checkpoint 输出绝对、pelvis-aligned 和
24 个身体部位误差：

```bash
python -m BodyPressure.pose_estimation.evaluate \
  --checkpoint runs/pressure_pose_transformer/best_model.pt \
  --data_root /home/shnh/DATA/zjy \
  --files BodyPressure/pose_estimation/val_files.txt \
  --device cuda
```

结果写到 checkpoint 同目录的 `evaluation.json`。手、腕、踝通常比躯干更能揭示模型是否真的
实现了“细粒度定位”；pelvis-aligned MPJPE 则区分全身平移误差与关节构型误差。
`evaluation.json` 还写入 dataset、synthetic/real domain、PI modality、split 文件、joint
convention 和 alignment，防止把不同协议下恰好接近的 MPJPE 误作横向排名。与 BodyMAP 论文
表格的详细可比性分析见 [`COMPARISON.md`](COMPARISON.md)。如果要比较 PVE、人体尺寸和
v2vP，先运行 `check_mesh_assets.py`；这些指标不能由 24 关节事后推导，需要 GT vertices、
GT per-vertex pressure、SMPL/SHAPY 和 EA indexes。

## 可替代数据的边界

1. **BodyPressureSD（首选预训练）**：公开且有 24 个 3D joints，最适合先打通细粒度定位；
   缺点是合成域，必须报告真实域差距。
2. **PmatData/PhysioNet**：公开压力数据适合姿态类别、呼吸或占床分析，但不应在未核对具体
   版本标注时宣称可直接监督 24 点 3D 姿态。
3. **自采少量真实集（首选微调/测试）**：使用顶视 RGB-D 或 mocap 仅在采集阶段生成伪标签，
   部署模型仍只读压力；按受试者划分并人工审核肩、肘、腕、髋、膝、踝。

如果目标是 SMPL mesh 而非关节定位，可在本模型的 joint tokens 后增加 SMPL 参数 head，
但那会重新引入受许可约束的 SMPL 文件；在没有真实 mesh 标签时不建议把它作为第一步。
