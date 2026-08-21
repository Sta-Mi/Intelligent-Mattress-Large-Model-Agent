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

### B. 取得合法真实测试数据后（最终论文应做）

把两种模型都放到同一 subject-disjoint 真实压力测试集，统一 SMPL-24 joint order、单位、
床坐标变换、root alignment 和样本权重。若使用 SLP，则必须按 BodyMAP 的 subjects 81--102、
45 poses 和三种 cover 条件评价；没有 SLP 时应自采真实 pressure + RGB-D/mocap 标注集。

## 关于 Pyramid Fusion

`RGB-D-PI-IR` 可以作为“多模态性能上界/参考”，但不能作为 pressure-only 的公平主基线。
论文表中的破折号还表示它没有报告 BodyMAP 的 mesh、shape 和 3D applied-pressure 指标，因此
即便 MPJPE 使用同一测试集，也只能比较共同报告的 pose 指标。
