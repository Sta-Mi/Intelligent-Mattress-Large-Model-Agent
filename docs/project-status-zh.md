# 智能床垫大模型智能体：当前状态与下一阶段

## 总目标

项目不是单一身份分类器，而是四个相互依赖的模块：

1. 压力图细粒度姿态/身体部位定位；
2. 单帧、序列和注册模板身份识别；
3. 压力、姿态、生理和环境时序的个性化睡眠质量评估；
4. 带安全约束的床垫分区自适应调节。

## 当前完成度

| 模块 | 当前状态 | 已有结果 | 是否完成 |
|---|---|---|---|
| 数据 | SLP cleaned pressure 已接入，101 人、45 姿态 | 五折 pose holdout 可复现 | 身份部分完成 |
| 身份 closed-set | PressureCNN + 五折分层 pose holdout | 多帧 subject accuracy `74.46% ± 4.72%` | 基线完成 |
| 身份 verification | ArcFace、P×K、SupCon、held-out pair/template evaluator | 五折模板 ROC-AUC `0.7959 ± 0.0190`，EER `0.2784 ± 0.0158` | 研究基线完成，未达部署 |
| 姿态估计 | 已下载 BodyMAP（Conv/PointNet/WS）代码与配置 | 尚未在当前环境形成训练指标表 | 未完成 |
| 睡眠质量 | 只有路线图，没有整夜标签读取器、时序模型或报告智能体 | 无 PSQI/PSG 对齐结果 | 未开始实现 |
| 自适应调节 | 只有路线图，没有执行器接口、仿真器或控制日志 | 无闭环指标 | 未开始实现 |

身份结果只能证明 SLP 压力图包含身份信号，不能代替姿态、睡眠质量和控制任务。模板
verification 的 EER 仍约 28%，只能称为研究原型，不能用于安全认证。

BodyMAP-PointNet `modality=both` 预训练权重已在 SLP real validation（22 人 × 45 姿态 × 3
遮被条件）完成推理：overall MPJPE `107.53 mm`、PVE `124.64 mm`、v2vP `3.307`。
uncover MPJPE/PVE 为 `93.56/108.75 mm`，cover1 为 `114.59/132.73 mm`，cover2 为
`114.44/132.44 mm`。该结果确认官方流程可运行，但仍需评估 depth-only 权重并与论文表格按
完全相同协议对齐后，才能判断是否复现官方精度。

同协议下 depth-only 预训练权重得到 overall MPJPE `59.45 mm`、PVE `71.78 mm`、v2vP
`2.473`，分别比当前 both checkpoint 低 `44.71%`、`42.41%`、`25.22%`。这是显著但反常的
负融合结果：在确认序列化模型自身的 `model.modality` 与配套 `exp.json` 一致、checkpoint
来源和训练 epoch 可比之前，只能表述为“当前两个预训练 checkpoint 的比较”，不能直接
得出“压力模态必然有害”的一般结论。

## 基线与公共数据

### 1. 细粒度姿态估计（现在的最高优先级）

主基线采用仓库已有的 **BodyMAP（CVPR 2024）**：

- 数据：SLP real + BodyPressureSD synthetic；
- 模型：BodyMAP-Conv、BodyMAP-PointNet，必要时比较 BodyMAP-WS；
- 输出：SMPL body mesh、3D joints、人体表面 applied pressure；
- 指标：MPJPE、PVE/V2V、pressure-map error、contact/part 指标；
- 源码与配置：`BodyPressure/BodyMAP/` 和 `model_config/*.json`。

BodyMAP 解决 3D mesh/pressure，不等同于精确 2D 身体部位定位。完成官方复现后，应在其
backbone 上增加 2D joint heatmap 与 body-part/contact segmentation head，并报告 PCK、IoU。

### 2. 身份识别（已形成第一版结果）

保留两条基线：

- PressureCNN softmax：固定已注册用户的 closed-set identification；
- ArcFace enrollment template：verification、新用户模板和阈值实验。

五折模板结果为：ROC-AUC `0.7959 ± 0.0190`、EER `0.2784 ± 0.0158`、
TAR@FAR=1% `0.2218 ± 0.0427`、subject accuracy `0.6614 ± 0.0815`。下一步只做必要的
姿态条件模板/序列聚合，不再把主要算力用于反复调单帧分类器。

### 3. 个性化睡眠质量评估

SLP 是短时受控姿态数据，不是整夜睡眠质量数据，不能从当前 45 帧直接训练“睡眠质量”。
需要两类数据：

- 压力侧：自行采集整夜压力序列，标注离床、翻身、姿态、高压持续时间；
- 生理/睡眠侧：Sleep-EDF、SHHS/MESA 等带 PSG/睡眠分期的数据用于时序睡眠表征预训练，
  但它们与 SLP pressure 不成对，不能假装是端到端多模态监督。

第一版模型应是“可验证时序模型 + 受约束大模型报告层”：

1. Pressure event encoder 提取姿态占比、翻身、离床、高压暴露；
2. PatchTST/TimesNet/Transformer 融合心率、呼吸、温湿度和事件序列；
3. 用真实睡眠分期、PSQI 或次日主观评分训练数值预测；
4. 大模型只读取结构化 JSON 证据并生成解释，不负责凭空计算医学分数。

### 4. 自适应床垫调节

在没有执行器日志和压力响应模型前，不直接训练在线强化学习。顺序必须是：

1. 定义动作：各气囊/分区软硬度、温度和最大变化率；
2. 规则控制器：限制峰值压力、持续高压和动作频率；
3. 学习床垫响应仿真器：`当前压力 + 姿态 + 动作 -> 下一压力/舒适度代理`；
4. 在仿真/离线日志中比较规则、监督策略、IQL/CQL；
5. 通过安全约束和人工确认后才允许小步在线调节。

## 接下来按此顺序执行

### M1：BodyMAP 官方复现

分别训练 Conv 和 PointNet 配置，保存 inference，再运行 metrics：

```bash
cd BodyPressure/BodyMAP/PMM
python main.py ../model_config/Conv.json
python main.py ../model_config/PointNet.json
```

产出统一表格：模型、输入模态、数据 split、MPJPE、PVE/V2V、pressure error、参数量、速度。

### M2：细粒度身体部位定位

在 M1 最优 backbone 上增加 joint heatmap + body-part/contact segmentation，多任务训练并输出
压力图上的肩、背、臀、腿等部位位置和置信度。

### M3：整夜事件与睡眠质量原型

先实现压力序列事件提取器和固定 JSON schema；没有整夜配对标签前，只做事件报告，不声称
预测真实睡眠质量。获得 PSQI/PSG/可穿戴对齐数据后，再训练数值时序模型和个性化校准。

### M4：安全调节仿真闭环

先实现规则控制与仿真器，评价峰值压力、持续高压时间、动作次数和主观舒适度；最后才进入
离线强化学习。

## 论文/报告必须分别给出的结果

1. 姿态：2D/3D joint、mesh、body-part/contact；
2. 身份：五折 closed-set、pair verification、template verification；
3. 睡眠：事件检测、睡眠分期/评分误差、解释证据一致性；
4. 控制：压力改善、动作安全、舒适度和消融实验；
5. 所有结果注明数据集、subject/session/pose split、随机种子和是否使用训练姿态注册模板。
