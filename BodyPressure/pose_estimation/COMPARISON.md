# PressurePoseTransformer 与 BodyMAP-PointNet PI：可比性说明

## 结论

如果最终系统只输入压力阵列，**传感器模态消融应主要与 `PI` 模型比较**，不能把
`RGB-D-PI-IR` 的 Pyramid Fusion 当作同输入基线。不过，“都标成 PI”只是必要条件，不是
充分条件；论文数值还必须使用同一数据集、测试对象、标签定义和对齐方式。

当前两个数字不能直接排名：

| 结果 | 输入 | 训练数据 | 测试数据 | 样本 | 输出 |
|---|---|---|---|---:|---|
| PressurePoseTransformer `77.93 mm` | PI | BodyPressureSD shape range 1--70 | BodyPressureSD shape range 71--80 | 12,381 | 直接回归 SMPL-24 joints |
| BodyMAP-PointNet PI `76.54±3.17 mm`（论文表） | PI（另传性别元数据） | BodyPressureSD 全部合成数据 + SLP real 训练 subjects | SLP real subjects 81--102 | 22×45×3 | SMPL mesh 派生 joints |

仓库代码可以验证协议差异：BodyMAP 的 `full-train-test` 使用全部 synthetic 文件和
`real_train.txt` 训练，用 `real_val.txt` 测试；`save_inference.py` 也固定读取 real validation。
此外训练和推理调用都把 ground-truth gender 两维标签传给模型，所以论文表中的 `PI` 指
传感器模态为 pressure image，不等于模型除了压力张量外没有任何元数据。

## 应如何报告

目前结果应写成：

> PressurePoseTransformer 在 BodyPressureSD synthetic、shape-range-disjoint（1--70/71--80）
> 协议上取得 77.93 mm absolute MPJPE；该结果不与 BodyMAP 在 SLP real test 上的 76.54 mm
> 做数值优劣比较。

同时报告 pelvis-aligned MPJPE `76.52 mm`。它只比 absolute MPJPE 低约 `1.41 mm`，而 pelvis
本身误差只有 `24.90 mm`，说明瓶颈不是全身平移，主要来自肢体构型；手、腕、足误差尤其高。

## 两种公平比较

### A. 无法取得 SLP 时（现在可做）

在完全相同的 BodyPressureSD 1--70/71--80 split 上重新训练并评价以下模型：

1. PressurePoseTransformer PI；
2. 一个 CNN/ResNet pressure-to-24-joint baseline；
3. BodyMAP-PointNet 的 `modality=pressure`，但只使用 synthetic 1--70 训练，禁止把 71--80
   纳入 `synth_all.txt`；评价同一批 12,381 个样本；
4. 分别给 BodyMAP 提供/不给 gender 元数据，作为独立消融。

共同指标只能先比较 absolute MPJPE 和 pelvis-aligned MPJPE。PressurePoseTransformer 不输出
6890 个 SMPL vertices，因此不能拿它与 BodyMAP 的 PVE、Height/Chest/Waist/Hips 或 v2vP
直接比较；要比较这些指标，必须增加 mesh/shape/pressure head 和相同 GT。

如果研究目标明确包含这些指标，主任务应升级为 BodyMAP 同定义的联合任务：

* PVE：输出 SMPL shape/pose/root，并通过同版本 SMPL 生成 6890 vertices；
* Height/Chest/Waist/Hips：对预测 vertices 使用相同的 SHAPY measurement pipeline；
* v2vP：输出每个 mesh vertex 的 applied pressure，并使用相同 EA1/EA2 vertex indexes；
* 把床垫 `64×27` pressure image 直接投影到 mesh 不等价于预测 v2vP。

先检查本地监督资产：

```bash
python -m BodyPressure.pose_estimation.check_mesh_assets \
  --data_root /home/shnh/DATA/zjy \
  --files BodyPressure/pose_estimation/val_files.txt \
  --smpl_root /path/to/BodyMAP/smpl_models/smpl \
  --out mesh_asset_audit.json
```

需要 `GT_BP_DATA/bp2/*_gt_vertices.npy`、`*_gt_pmaps.npy`、男女 SMPL 模型和 parsed EA
indexes。只有 `synth` 与 `synth_depth` 仍不足以计算这些指标。

资产检查通过后，仓库提供两个相同 split、相同 BodyMAP mesh/PME16 decoder 的配置：

```bash
cd BodyPressure/BodyMAP/PMM
python main.py ../model_config/PointNetPressureSynth.json
python main.py ../model_config/PressureTransformerSynth.json
```

第一项是官方 ResNet-18 BodyMAP-PointNet PI backbone，第二项是 ViT pressure backbone；两者
共享 BodyMAP 的 SMPL 与 PME16 decoder，只用 BodyPressureSD 1--70 训练，并在 71--80
validation 上计算 MPJPE、PVE、Height/Chest/Waist/Hips 和 v2vP/1EA/2EA。训练期详细 metric
每 100 epoch 运行一次，因为 mesh 截面人体尺寸计算明显慢于 loss validation。

纯 synthetic 配置的 inference loader 会跳过 `real_file=None`，不会再尝试把 `None` 传给
`open()`。启动日志中的 SMPL NumPy non-writable warning 来自上游 `smplpytorch` 将只读 Chumpy
数组包装为 tensor；当前代码没有写该 tensor，它不是此次数据加载失败的原因。

外层 tqdm 以 epoch 为单位，85,114 个样本、batch size 32 时每个 epoch 有约 2,660 个
training batches；旧日志会在完整 epoch 结束前一直显示 `0/100`，不能据此判断 GPU 卡死。新版
默认显示 batch 进度和 loss，并启用 CUDA AMP。先用两批 smoke config 验证 forward/backward：

```bash
python main.py ../model_config/PressureTransformerSynthSmoke.json
```

该配置使用 batch size 8，只运行两个 train batches，跳过完整 validation 和最终 mesh metrics；
通过后再运行 `PressureTransformerSynth.json`。正式配置每 5 epoch validation 一次，以避免每个
epoch 都额外遍历 12,381 个 mesh validation samples。

外层显示如 `[13:32<22:19:59, 812.12s/it]` 表示一个 epoch（含当轮 validation）约 13.5
分钟，而不是一个 batch 需要 812 秒；对应 2,660 个 train batches 时约 0.31 秒/batch，属于
PMM7 + 6890 vertices + PME16 的正常 GPU 吞吐。新版只在第 5、10、… epoch 做 validation，
并在每轮打印分钟数、秒/batch 和粗略剩余小时数；validation epoch 的 naive ETA 会偏高。

这里必须区分“一个 epoch”和“一次完整训练 run”：上述 tqdm 的总数是 `100 epochs`，所以
`22:19:59` 是剩余 99 epochs 的总 ETA，**不是一个 epoch 要 22 小时**。按 812 秒/epoch，单个
epoch 是 13.54 分钟，100 epochs 的完整 run 约 22.56 小时。若实际观察到 epoch 1 自身运行了
20 小时，则日志应接近 `72000s/it`，与 `812.12s/it` 不是同一种情况；此时应保存完整的
`train batches` 行和 `nvidia-smi` 利用率再排查。

例如 `21/100 [4:42:43<17:43:45, 807.91s/it]` 表示完整训练实验的 100 epochs 已完成 21
epochs：每个 epoch 约 807.91 秒（13.47 分钟），已经训练 4 小时 42 分，剩余 79 epochs 估计
17 小时 43 分。新版外层进度条显式标为 `epochs (complete training run)` 且单位为 `epoch`。
若启动行仍显示 `Validation every 1 epoch(s)`，说明运行机器使用的配置尚未更新到仓库当前
`epochs_validate=5`，并非进度条含义发生变化。

### B. 取得合法真实测试数据后（最终论文应做）

把两种模型都放到同一 subject-disjoint 真实压力测试集，统一 SMPL-24 joint order、单位、
床坐标变换、root alignment 和样本权重。若使用 SLP，则必须按 BodyMAP 的 subjects 81--102、
45 poses 和三种 cover 条件评价；没有 SLP 时应自采真实 pressure + RGB-D/mocap 标注集。

## 关于 Pyramid Fusion

`RGB-D-PI-IR` 可以作为“多模态性能上界/参考”，但不能作为 pressure-only 的公平主基线。
论文表中的破折号还表示它没有报告 BodyMAP 的 mesh、shape 和 3D applied-pressure 指标，因此
即便 MPJPE 使用同一测试集，也只能比较共同报告的 pose 指标。

## 100 epochs 结束后的操作

训练进度到 `100/100` 后，trainer 还会执行 final inference 和完整 synthetic validation metric；
人体截面与 6890-vertex pressure metric 可能继续运行较长时间。应等到终端打印
`Final metrics saved to .../metrics.json` 并返回 shell prompt。结果目录为：

```text
PMM_exps/normal/exps/PressureTransformerMesh_SynthSplit/
```

随后用同一协议训练 BodyMAP PI 基线：

```bash
cd BodyPressure/BodyMAP/PMM
python main.py ../model_config/PointNetPressureSynth.json
```

两个 run 都完成后生成 Markdown 比较表和机器可读 JSON：

```bash
cd BodyPressure/BodyMAP
python scripts/compare_synth_metrics.py \
  Transformer=../../PMM_exps/normal/exps/PressureTransformerMesh_SynthSplit/metrics.json \
  BodyMAP-PI=../../PMM_exps/normal/exps/BodyMAP_PointNet_PI_SynthSplit/metrics.json \
  --out ../../PMM_exps/normal/pressure_synth_comparison.json
```

比较器会拒绝非 `pressure` modality 或 overall sample count 不是 12,381 的结果，避免再次混用
SLP real 表格。主要报告 `overall`，并保留 `f`/`m` 分组作为性别差异诊断。旧 run 若没有独立
`metrics.json`，可把命令中的文件改成同目录 `exp.json`；比较器兼容其中的 `metric` 字段。

### 当前 PMM7 100-epoch 结果

在 12,381 个 BodyPressureSD 71--80 synthetic validation 样本上，PMM7 得到：

| MPJPE mm | PVE mm | Height cm | Chest cm | Waist cm | Hips cm | v2vP | 1EA | 2EA |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 136.256 | 154.210 | 0.665 | 18.010 | 8.882 | 8.556 | 1.689 | 1.138 | 0.837 |

`synth` 与 `overall` 相同是预期行为，因为本次 evaluator 只有 synthetic 数据；`uncover`、
`cover1`、`cover2` 的 `count=0/null` 也不是错误。female/male counts 为 6,252/6,129，总和
等于 12,381。男性相较女性的 MPJPE/PVE 分别高约 6.00/6.92 mm，v2vP 高约 0.196，需作为
分组差异报告，但不能在没有 BMI/体型分层前归因于性别本身。

该 MPJPE 明显高于 joint-only PressurePoseTransformer 的 77.93 mm，但二者输出和损失不同：
PMM7 同时回归 SMPL mesh 与 vertex pressure，不能据此认定 Transformer encoder 本身更差。
下一项必须运行 `PointNetPressureSynth.json`；只有它与 PMM7 使用相同 SMPL/PME16、多任务损失、
split 和 evaluator，才能回答 PMM7 是否优于 BodyMAP-PointNet PI。论文 SLP PI 的 76.54 mm
仍不可放进此表做数值排名。
