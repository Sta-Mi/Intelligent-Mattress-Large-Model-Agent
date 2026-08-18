# 智能床垫大模型智能体：分步实施路线图


## 0. 基线选择原则：不是简单选四个模型

结论：**需要为四个子任务分别设置基线，但不要理解成只选 4 个单模型**。更合理的做法是给每个子任务设置“保底可复现基线 + 最前沿主基线 + 消融/替代基线”。论文主线可以突出 4 个最前沿主基线，工程上至少保留传统或轻量基线，便于证明先进模型确实带来增益。

| 子任务 | 推荐最前沿主基线 | 为什么选它 | 保底可复现基线 |
|---|---|---|---|
| 细粒度姿态估计 | **BodyMAP / BodyMAP-WS 思路**：压力图 + 深度/多模态输入，联合预测人体 mesh 与 3D applied pressure map | 截至当前调研，BodyMAP 面向“人在床上”的 3D 人体网格和全身 3D 压力图联合预测，任务最贴近智能床垫细粒度身体部位定位 | HRNet/ConvNeXt/Swin + heatmap 关键点回归 |
| 身份识别 | **Swin/ConvNeXt/ViT + ArcFace 或 SupCon 度量学习**，最好扩展为压力序列 Transformer | 压力身份识别没有统一公认 foundation baseline，最稳妥是用强视觉 backbone 学压力纹理，再用度量学习支持 closed-set 与 open-set 识别 | ResNet-50 分类器、TCN/Transformer 序列分类器 |
| 个性化睡眠质量评估 | **睡眠/时序 foundation model + LLM agent**：时序模型抽取夜间事件，大模型负责个性化解释和建议 | 睡眠质量不是单帧识别，而是长时序、多模态、个体化推理；应把压力事件、用户画像、可穿戴/环境数据融合到结构化证据后再交给大模型 | PatchTST/TimesNet/Informer + 规则评分 |
| 自适应床垫调节 | **安全离线强化学习**：C2IQL / IQL / CQL 类方法 + 硬安全约束 | 床垫调节不能直接在线乱试，必须从历史日志或仿真器学习，并用约束保证动作安全 | 规则控制器、MPC、监督学习 policy |

选型策略建议：论文或项目汇报中把上述 4 个“推荐最前沿主基线”作为核心 baseline；实际实验中每个任务再配 1-2 个弱基线，形成从传统方法到先进方法的对照。


## 0.1 四个方向的可复现基线、论文与公开仓库

| 方向 | 最推荐的可复现基线 | 对应论文 | 公开仓库/代码状态 | 备注 |
|---|---|---|---|---|
| 细粒度姿态估计 | **BodyMAP / BodyMAP-WS** | Tandon et al., **BodyMAP: Jointly Predicting Body Mesh and 3D Applied Pressure Map for People in Bed**, CVPR 2024 | 官方仓库：`https://github.com/rchi-lab/bodymap` | 最贴近本项目，因为它直接面向人在床上的 body mesh 与 3D applied pressure map 联合预测；输入包含深度图和 2D 压力图。 |
| 细粒度姿态估计的保底复现 | **SLP + stacked hourglass / in-bed pose estimation baseline** | Liu et al., **Simultaneously-Collected Multimodal Lying Pose Dataset: Enabling In-Bed Human Pose Monitoring**, TPAMI 2023；以及 Seeing Under the Cover / SLP 相关代码 | 官方/作者仓库：`https://github.com/ostadabbas/SLP-Dataset-and-Code`、`https://github.com/ostadabbas/Seeing-Under-the-Cover`、`https://github.com/ostadabbas/in-bed-pose-estimation` | 如果 BodyMAP 环境较重，可先在 SLP 上复现 2D pose/heatmap，再升级到 mesh。 |
| 身份识别 | **Deep multi-branch CNN for pressure-map subject/posture recognition** | Davoodnia et al., **Identity and Posture Recognition in Smart Beds with Deep Multitask Learning**, 2021 | 未找到官方代码；可复现数据来自 PhysioNet PmatData：`https://physionet.org/content/pmd/`；可参考社区复现：`https://github.com/Fustincho/UD-Private-In-bed-Posture-Classification` | 这是压力床垫身份识别最直接的可复现论文线；若追求更先进，可把 backbone 换成 ConvNeXt/Swin/ViT，并加入 ArcFace/SupCon。 |
| 身份识别的先进替代 | **ArcFace-style metric learning on pressure embeddings** | Deng et al., **ArcFace: Additive Angular Margin Loss for Deep Face Recognition**, CVPR 2019 | 开源实现：`https://github.com/deepinsight/insightface` | ArcFace 原始任务是人脸识别，不是床垫压力身份识别；这里是把其角度间隔度量学习思想迁移到压力 embedding。 |
| 个性化睡眠质量评估 | **SleepFM + 结构化压力事件 + LLM agent** | Thapa et al., **SleepFM: Multi-Modal Sleep Foundation Model**, 2024；Nature Medicine 2026 扩展版为 multimodal sleep foundation model for disease prediction | 代码仓库：`https://github.com/rthapa84/sleepfm-codebase`、临床扩展：`https://github.com/zou-group/sleepfm-clinical` | SleepFM 使用 PSG 多模态信号，不直接使用床垫压力图；本项目应把压力图转换为夜间事件/时序特征，再与 SleepFM/LLM 融合。 |
| 个性化睡眠质量评估的通用时序替代 | **MOMENT time-series foundation model** | Goswami et al., **MOMENT: A Family of Open Time-Series Foundation Models**, ICML 2024 | 官方仓库：`https://github.com/moment-timeseries-foundation-model/moment` | 如果没有 PSG 标签，MOMENT 更适合先做压力时序表示学习、分类、异常检测和少样本微调。 |
| 自适应床垫调节 | **IQL / CQL 离线强化学习 + 安全规则约束** | Kostrikov et al., **Offline Reinforcement Learning with Implicit Q-Learning**, 2021；Kumar et al., **Conservative Q-Learning for Offline Reinforcement Learning**, NeurIPS 2020 | IQL 官方仓库：`https://github.com/ikostrikov/implicit_q_learning`；CQL 官方仓库：`https://github.com/aviralkumar2907/CQL` | 这是可复现、成熟的离线 RL 基线；床垫调节必须先离线训练和仿真验证，不能直接在线探索。 |
| 自适应床垫调节的前沿安全替代 | **C2IQL / FISOR safe offline RL** | Liu et al., **C2IQL: Constraint-Conditioned Implicit Q-learning for Safe Offline Reinforcement Learning**, ICML 2025；Zheng et al., **Safe Offline Reinforcement Learning with Feasibility-Guided Diffusion Model**, ICLR 2024 | C2IQL 论文公开，未确认官方代码；FISOR 官方仓库：`https://github.com/ZhengYinan-AIR/FISOR` | 作为前沿安全控制方向；如果代码可用性优先，先用 IQL/CQL，再把安全约束迁移到 C2IQL/FISOR。 |

推荐最终写法：**四个方向各有一个主 baseline**，但其中姿态估计和睡眠质量评估可以选择“领域专用公开模型”，身份识别和调节控制则更适合采用“公开通用先进算法 + 压力床垫数据重训练”的方式。


## 0.2 最快实现智能体：必须下载的数据集与必须复现的代码

如果目标是**尽快做出一个先进模型驱动的智能床垫智能体原型**，不要一开始同时铺开所有数据和所有论文。建议按下面的优先级执行。

### 第一优先级：必须下载的数据集

| 优先级 | 数据集 | 下载/入口 | 先解决哪个任务 | 为什么必须先下 |
|---|---|---|---|---|
| P0 | **PmatData / PhysioNet Pressure Map Dataset** | `https://physionet.org/content/pmd/` | 姿态分类、身份识别、压力图数据管线 | 最轻量、最容易下载和跑通；适合先把压力图读取、归一化、划分、训练、评估流程打通。 |
| P0 | **SLP Dataset** | `https://github.com/ostadabbas/SLP-Dataset-and-Code` | 细粒度姿态估计、多模态对齐 | 有 RGB、LWIR、Depth、Pressure Map 和遮被条件，是做“压力图到姿态”的关键公开数据集。 |
| P0 | **BodyMAP / BodyPressureSD 相关数据** | `https://github.com/rchi-lab/bodymap` | 3D body mesh、3D pressure map、先进姿态估计 | 这是最贴近“智能床垫压力分布 + 细粒度身体部位定位”的先进路线。 |
| P1 | **SleepFM 使用的 PSG/睡眠信号数据入口** | `https://github.com/rthapa84/sleepfm-codebase` | 个性化睡眠质量评估 | 床垫压力图本身通常没有真实睡眠分期/睡眠质量标签；SleepFM 可提供睡眠 foundation model 表示能力。 |

### 第一优先级：必须复现的代码

| 优先级 | 先复现什么代码 | 仓库 | 预期产出 |
|---|---|---|---|
| P0 | **PmatData 姿态分类 + 身份识别小基线** | 数据来自 `https://physionet.org/content/pmd/`；可自写 PyTorch/Lightning 小模型，或参考社区压力姿态分类仓库 | 1-2 天内得到可运行的数据读取器、训练脚本、accuracy/EER/混淆矩阵。 |
| P0 | **SLP pressure map 可视化和读取接口** | `https://github.com/ostadabbas/SLP-Dataset-and-Code` | 证明能正确读取 PM、RGB、Depth、LWIR，并完成跨模态样本对齐。 |
| P0 | **BodyMAP 官方训练/推理流程** | `https://github.com/rchi-lab/bodymap` | 得到最先进姿态估计 baseline：body mesh + 3D pressure map。 |
| P1 | **SleepFM embedding / downstream 示例** | `https://github.com/rthapa84/sleepfm-codebase` | 得到睡眠 foundation model 表示；后续把床垫事件摘要接入 LLM agent。 |
| P1 | **IQL 或 CQL 离线 RL 最小示例** | `https://github.com/ikostrikov/implicit_q_learning`、`https://github.com/aviralkumar2907/CQL` | 先在标准 offline RL benchmark 跑通，再替换为床垫仿真环境。 |

### 最快落地顺序

1. **第 1 步：先做 PmatData**。目标不是最先进，而是最快把压力图数据管线跑通：下载、解析、归一化、划分、训练 CNN、输出姿态分类和身份识别结果。
2. **第 2 步：再做 SLP**。目标是把压力图和多模态姿态标注对齐，复现 pressure map pose baseline。
3. **第 3 步：复现 BodyMAP**。这是姿态估计方向最应该优先追的先进 baseline；跑通后再考虑把输入改成“只用压力图”或“压力图 + 少量传感器”。
4. **第 4 步：睡眠质量先做事件流，不要直接训练大模型**。先从压力序列提取翻身、离床、姿态占比、局部高压持续时间，再调用 SleepFM/MOMENT/LLM 做解释。
5. **第 5 步：自适应调节先做规则控制器**。没有真实床垫调节日志前，不要直接上 RL；先用规则控制器和仿真器，再接 IQL/CQL。

### 如果只能选 3 个最关键资源

1. **PmatData**：最快跑通压力图姿态分类和身份识别。
2. **SLP Dataset and Code**：最快接入多模态姿态估计监督。
3. **BodyMAP**：最先进、最贴近“床垫压力 + 细粒度人体定位”的姿态估计 baseline。

SleepFM 和 IQL/CQL 可以作为第二阶段：前者用于睡眠质量解释，后者用于自适应调节闭环。

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
