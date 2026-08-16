# 智能床垫大模型智能体：分步实施路线图

## 1. 目标拆解

本项目建议拆成四条可并行但逐步集成的技术线：

1. **压力分布细粒度姿态估计**：输入智能床垫压力图，输出 2D/3D 关键点、身体部位热力图、SMPL/SMPL-X 人体网格或接触压力区域。
2. **身份识别**：输入单帧或睡眠片段压力序列，区分不同个体；需要严格避免训练/测试同床垫、同姿态泄漏造成虚高结果。
3. **个性化睡眠质量评估**：融合压力、姿态、翻身频率、离床、心率/呼吸/温湿度等多模态数据，用多模态大模型或时序基础模型生成可解释评分和建议。
4. **自适应床垫调节**：基于姿态、压力峰值、历史偏好和睡眠阶段，学习分区气囊/软硬度/温度的调节策略，并用安全约束限制动作幅度。

## 2. 文献与公共数据集优先级

建议优先复现和使用下列公开资源：

| 优先级 | 数据集/论文 | 适合任务 | 关键说明 |
|---|---|---|---|
| P0 | **SLP: Simultaneously-Collected Multimodal Lying Pose** | 多模态姿态估计、遮被鲁棒性、跨模态学习 | 包含 RGB、LWIR、深度、压力图和不同遮被条件；适合把压力图与视觉/深度模态对齐。 |
| P0 | **BodyPressure / BodyPressureSD** | 3D 人体网格、接触压力、合成到真实迁移 | 提供大量合成 SMPL 姿态和压力图，可用于预训练或增强真实数据。 |
| P0 | **PmatData / PhysioNet Pressure Map Dataset** | 姿态分类、身份识别 | 经典公开压力床垫数据集，包含多被试、多睡姿，适合作为身份识别与姿态分类基线。 |
| P1 | **Pressure-Sensing Mat Dataset** | 3D 关节点姿态估计 | 含压力图与 3D 人体姿态标注，可作为细粒度姿态估计补充。 |
| P1 | **TIP / temporal pressure datasets** | 睡眠过程时序建模、动态姿态 | 如果可获取，适合训练翻身、转移、长期睡眠状态建模。 |

参考入口：SLP 论文和代码仓库、BodyPressure 仓库、PhysioNet PmatData 页面，以及压力图数据集综述应作为第一轮调研材料。

## 3. 基线模型选择

### 3.1 细粒度姿态估计基线

1. **第一基线：压力图到关键点热力图**
   - Backbone：ConvNeXt、Swin Transformer 或 HRNet。
   - Head：关键点 heatmap + body-part segmentation + contact map 多任务输出。
   - 指标：PCK、MPJPE、关键点 heatmap MSE、身体部位 IoU。

2. **前沿基线：压力图到 3D 人体网格**
   - Backbone：ViT/Swin/ConvNeXt + token decoder。
   - Output：SMPL/SMPL-X 参数、3D joints、接触压力分布。
   - 训练策略：先用 BodyPressureSD 合成数据预训练，再在 SLP/Pmat 等真实数据上微调。

3. **跨模态蒸馏基线**
   - Teacher：RGB/Depth/LWIR 上的成熟 2D/3D pose estimator。
   - Student：只输入压力图，学习 teacher 的关键点、人体区域和姿态 embedding。
   - 优点：部署时只用压力图，训练时利用多模态监督。

### 3.2 身份识别基线

1. **单帧分类基线**：ResNet/ConvNeXt/Swin 输入压力图，输出 subject ID。
2. **序列识别基线**：TCN、Transformer Encoder 或 TimeSformer 输入多帧压力序列。
3. **度量学习基线**：ArcFace / SupCon / Triplet loss，评估 closed-set accuracy 和 open-set EER。身份识别必须做 subject-disjoint 或 session-disjoint 划分，避免同一人的近邻帧同时进入训练和测试。

### 3.3 个性化睡眠质量评估基线

1. **特征层**：姿态比例、翻身次数、离床事件、局部高压持续时间、睡眠连续性、传感器统计特征。
2. **时序模型**：PatchTST、TimesNet、Informer 或 Transformer Encoder。
3. **多模态大模型层**：将结构化时序特征、姿态事件摘要和用户画像转成文本/JSON，由多模态或通用大模型生成个性化解释、风险提示和调节建议。
4. **评估**：如果没有真实 PSQI/睡眠分期标签，先做弱监督规则评分；后续采集问卷、可穿戴设备或 PSG 子集做校准。

### 3.4 自适应床垫调节基线

1. **安全规则控制器**：先建立不可超过压力、角度、温度、气囊变化率阈值的规则系统。
2. **监督学习策略**：用专家规则或历史人工调节记录训练 policy network。
3. **离线强化学习**：在仿真环境或历史日志上训练 Conservative Q-Learning / IQL 类方法。
4. **在线闭环**：小步长 A/B 测试，优先优化局部高压持续时间、翻身次数、主观舒适度，不直接追求激进控制。

## 4. 推荐实施步骤

### 阶段 0：定义协议

- 明确传感器规格：压力图分辨率、采样率、压力单位、床垫分区控制接口。
- 明确任务输出：姿态类别、2D/3D 关键点、身份 ID、睡眠质量评分、床垫调节动作。
- 固定数据划分：按被试、日期、床垫设备划分 train/val/test。

### 阶段 1：复现公开数据集基线

1. 下载并整理 SLP、BodyPressureSD、PmatData。
2. 写统一数据读取器，把所有压力图转成统一张量格式：`C x H x W` 或 `T x C x H x W`。
3. 先复现三个最小实验：
   - PmatData 姿态分类。
   - PmatData 身份识别。
   - SLP 或 Pressure-Sensing Mat 的压力图关键点估计。
4. 建立实验日志：数据版本、划分、模型、指标、随机种子。

### 阶段 2：训练前沿姿态模型

1. 用 BodyPressureSD 预训练 3D pose/SMPL 模型。
2. 用 SLP 压力图与 RGB/Depth/LWIR 模态做蒸馏或对比学习。
3. 加入多任务损失：关键点、人体部位、接触区域、姿态类别。
4. 做跨被试测试，重点报告遮被条件、不同 BMI/身高/体重分组表现。

### 阶段 3：身份识别与隐私评估

1. 构造 closed-set 与 open-set 两套协议。
2. 比较单帧 CNN、序列 Transformer、ArcFace embedding。
3. 分析身份泄漏风险：姿态是否被模型错误当作身份特征，训练/测试是否有相邻帧泄漏。
4. 输出隐私策略：本地推理、匿名 embedding、可撤销用户模板。

### 阶段 4：睡眠质量大模型智能体

1. 将一晚压力序列转换为事件流：入睡、离床、翻身、长时间高压、姿态转换。
2. 构造用户画像：身高体重、偏好软硬度、病史限制、历史睡眠反馈。
3. 让大模型只基于结构化证据生成解释，避免凭空诊断。
4. 输出固定 JSON：`score`、`risk_factors`、`evidence`、`recommendations`、`adjustment_plan`。

### 阶段 5：自适应调节闭环

1. 从规则控制器开始，保证动作安全可解释。
2. 建立床垫调节仿真器：输入当前姿态和动作，预测压力图变化与舒适度代理指标。
3. 用离线数据训练 policy，再用人类反馈微调。
4. 上线时只允许小幅、低频调节，并保留用户一键关闭与回滚机制。

## 5. 最小可行里程碑

1. **第 1-2 周**：完成文献表、数据下载脚本、统一数据格式。
2. **第 3-4 周**：复现 PmatData 姿态分类和身份识别基线。
3. **第 5-8 周**：完成 SLP/BodyPressure 的关键点或 SMPL 姿态估计训练。
4. **第 9-10 周**：完成睡眠事件提取和大模型报告原型。
5. **第 11-12 周**：完成规则调节器和离线仿真验证。

## 6. 第一版仓库结构建议

```text
data/                 # 不提交原始大数据，只放说明和索引
configs/              # 数据集、模型、训练配置
src/datasets/         # SLP、PmatData、BodyPressure 读取器
src/models/           # pose、identity、sleep-quality、control 模型
src/training/         # 训练和评估脚本
src/agents/           # 睡眠质量评估与调节智能体
experiments/          # 实验配置快照和结果表
reports/              # 文献综述、指标、可视化结果
```
