# FACET-0 复现开发计划

> 版本：v1.0  
> 日期：2026-09-03  
> 工作目录：`/home/hzm/code/facet_0`  
> 目标：先使官方 FACET-0 checkpoint 完成可验证的离线推理，再逐步近似复现论文中的 action-wrench 联合学习、价值引导强化学习和局部适配。

## 执行进度

- [x] 阶段 0：建立可复现环境（2026-09-03）
- [x] 阶段 1：checkpoint 取证与兼容配置（2026-09-03）
- [x] 阶段 2：FACET 输入输出适配层（2026-09-03；相机语义映射待官方确认）
- [x] 阶段 3：官方 checkpoint 离线推理（2026-09-03；100/100 样本通过）
- [x] 阶段 4：数据质量与离线评估基线（2026-09-03；全量视频码流仅抽样解码）
- [ ] 阶段 5～10：训练扩展、真机与论文级评估

当前验证结果：FACET checkpoint 的 51 个参数叶节点与标准 OpenPI π0.5 配置逐项完全匹配，且已在 RTX 5090 上完成无裁剪恢复。详见 `reports/openpi_compatibility.md`。

## 1. 结论与范围

本项目按三个复现层级推进，避免把“加载官方权重”“重写论文算法”和“复现论文真机指标”混为一件事。

| 层级 | 目标 | 当前可行性 | 本计划处理方式 |
|---|---|---:|---|
| L1：推理复现 | 加载已发布 checkpoint，输入图像、状态和指令，输出动作块 | 高 | 第一优先级，先形成可运行基线 |
| L2：算法复现 | 在公开数据上实现论文描述的联合 action-wrench、VQA、critic 和 TD3 | 中 | 分模块实现，并明确标记推测项 |
| L3：结果复现 | 复现论文中的五项真机任务、82% 平均成功率和亚毫米精度 | 低 | 需要完整数据、真机、控制系统及未公开实现细节 |

第一里程碑不要求真机：使用真实数据集样本完成端到端离线推理，证明 checkpoint、预处理、归一化和动作解码之间相互兼容。

## 2. 当前已有资产与约束

### 2.1 已有资产

- 论文：`Facet-0.pdf`，26 页。
- 模型：`Facet-0/facet0-post-training/`，约 11.58 GiB。
- 数据：`ManuFacet-1K/Facet0-1/`，约 23.43 GiB。
- 数据规模：2,622 episodes、2,702,153 frames、7,866 个视频、约 50.04 小时、20 个任务。
- 官方基础框架：[Physical Intelligence OpenPI](https://github.com/Physical-Intelligence/openpi)。
- FACET 官方代码仓库：[PINE-Lab-NTU/FACET](https://github.com/PINE-Lab-NTU/FACET)，截至本计划编写时仍标注 `Code coming soon`。

### 2.2 已确认的模型接口

- 模型类型：OpenPI 风格 vision-language-action model。
- 视觉输入：三路 RGB 图像。
- 状态输入：13 维，即末端位姿与夹爪 7 维，加六轴力/力矩 6 维。
- 动作：绝对末端位姿与夹爪，共 7 维。
- 动作块长度：50。
- 夹爪约定：`1 = closed`，`0 = open`。
- checkpoint：JAX/Orbax OCDBT 格式。
- 参数结构：与 π0.5 的 PaliGemma、action projection、time MLP 结构相符。
- 归一化统计：state 前 13 维有效，action 前 7 维有效，其余维度补零至 32 维。

### 2.3 当前计算资源

- GPU：NVIDIA GeForce RTX 5090，32 GB 显存。
- Python：系统 Python 3.12.3；实际环境应由 `uv` 使用 OpenPI 锁定版本，不直接依赖系统解释器。
- 磁盘：当前分区剩余约 1.1 TB。

资源判断：足够完成 JAX/PyTorch 推理、数据检查、轻量训练和大多数 LoRA 实验；完整参数微调预计需要 70 GB 以上显存，应改用多卡 FSDP 或 80 GB GPU。

## 3. 核心工程原则

1. 保留原始模型和数据目录，只读使用，不修改 checkpoint 与公开数据。
2. 先验证官方权重，再实现论文扩展；任何未验证模块不能污染推理基线。
3. 固定 OpenPI commit、Python 和依赖版本，生成完整环境锁文件。
4. 数据适配集中在 transforms/adapter 层，不散落到模型代码中。
5. 对论文未公开参数使用显式配置，并标注 `paper`、`inferred` 或 `experimental` 来源。
6. 所有训练先通过小数据 overfit 和离线单元测试，再投入长时间计算。
7. 真机控制必须设置工作空间、单步位移、姿态、速度、夹爪和力矩安全边界。

## 4. 建议项目结构

```text
facet_0/
├── Facet-0.pdf
├── Facet-0/                       # 官方权重，只读
├── ManuFacet-1K/                  # 官方数据，只读
├── third_party/
│   └── openpi/                    # 固定 commit 的上游代码
├── src/facet0/
│   ├── configs/                   # 模型、数据、训练与推理配置
│   ├── data/                      # ManuFacet adapter、视频解码、采样
│   ├── models/                    # checkpoint wrapper 与论文扩展模块
│   ├── policies/                  # 动作推理、CFG、chunk 执行策略
│   ├── training/                  # 联合训练、critic、TD3
│   ├── evaluation/                # 离线指标与真机评估协议
│   └── robot/                     # 真机接口与安全控制，后期建立
├── scripts/                       # 可复现命令入口
├── tests/                         # 单元、集成和回归测试
├── configs/                       # YAML/TOML 实验配置
├── reports/                       # 检查报告、指标与曲线
└── artifacts/                     # 本地产物，不提交大文件
```

## 5. 分阶段开发路线

工期为单名熟悉 JAX/PyTorch 的工程师估算，不包括长时间训练、数据采集和真机排期。

### 阶段 0：建立可复现环境

预计工期：0.5～1 个工程日。

任务：

- 克隆官方 OpenPI，并固定到经过验证的 commit。
- 使用 `uv` 建立独立环境，优先运行 JAX 原生 checkpoint，PyTorch 作为后续可选路径。
- 记录 CUDA driver、JAX、Orbax、Flax、Transformers 和 OpenPI 版本。
- 增加显存检查、设备枚举和最小 OpenPI smoke test。
- 设置缓存目录和 `.gitignore`，避免重复下载及误提交大文件。

交付物：

- `pyproject.toml`、`uv.lock` 或上游环境锁定说明。
- `scripts/check_environment.py`。
- `reports/environment.json`。

验收标准：

- OpenPI 能识别 RTX 5090。
- JAX 完成一次矩阵计算且无 CUDA/算子错误。
- 环境可从空虚拟环境按文档重新创建。

### 阶段 1：checkpoint 取证与兼容配置

预计工期：1～2 个工程日。

任务：

- 使用 Orbax 读取完整 global shape、dtype、sharding 和参数路径。
- 将参数树与 OpenPI `Pi0Config(pi05=True)` 的期望树逐叶比较。
- 确认以下关键配置：
  - `action_dim = 32`（其中 FACET 实际动作使用前 7 维）；
  - `action_horizon = 50`；
  - PaliGemma 和 action expert variant；
  - token 最大长度；
  - checkpoint 是否能从原 2-device mesh 重分片到本机单卡。
- 区分 checkpoint 中字段名为 `value` 的数组包装节点与论文的 value network，避免错误判断。
- 生成参数兼容性报告和模型参数量统计。

交付物：

- `scripts/inspect_checkpoint.py`。
- `reports/checkpoint_inventory.json`。
- `reports/openpi_compatibility.md`。
- FACET 专用 OpenPI model config。

验收标准：

- 参数树不存在未解释的 shape mismatch。
- checkpoint 可恢复到 CPU 内存，并可进一步恢复到 GPU。
- 明确列出 checkpoint 包含和不包含的论文模块。

决策门：如果上游最新版无法加载，则优先根据 checkpoint 元数据寻找相容的历史 OpenPI commit；不直接修改权重文件。

### 阶段 2：FACET 输入输出适配层

预计工期：2～3 个工程日。

任务：

- 建立三路相机映射：
  - `observation.images.view1`；
  - `observation.images.hand`；
  - `observation.images.view2`。
- 明确三路图像分别映射到 OpenPI 的 base/left-wrist/right-wrist 占位键；映射作为配置项，不硬编码语义。
- 将图像从 640×480 解码、裁剪/缩放到 OpenPI 的 224×224 输入，并验证 RGB 顺序和数值范围。
- 将 13 维状态按 `[x,y,z,rx,ry,rz,gripper,fx,fy,fz,tx,ty,tz]` 拼接并补零到 32 维。
- 将 7 维动作补零至 32 维，推理后只解码前 7 维。
- 实现 norm/unnorm，处理无效补零维，禁止对零方差执行除法。
- 从 episode/task 元数据生成自然语言 prompt；保留原始 subtask 与 task id 便于追溯。

交付物：

- `src/facet0/data/manufacet.py`。
- `src/facet0/data/transforms.py`。
- `src/facet0/policies/facet_policy.py`。
- 数据 schema 和映射文档。

验收标准：

- 从任意有效 episode 读取连续 50 帧时，图像、状态、动作和 prompt shape 全部正确。
- norm → unnorm 往返误差小于 `1e-5`。
- 所有张量均为有限值，mask 与 padding 正确。
- 同一 episode 的三路视频时间戳对齐误差不超过一帧。

### 阶段 3：官方 checkpoint 离线推理

预计工期：1～3 个工程日。

任务：

- 建立统一推理 CLI，支持 dataset episode/frame、单张输入包和随机 smoke input。
- 在 JAX 下执行 flow-matching action sampling。
- 查明并实现发布配置中的 conditional/unconditional guidance；若标准 OpenPI checkpoint 不支持，先将 `guidance_scale=1.0` 作为基线，并在报告中说明。
- 固定随机种子、采样步数和数值精度。
- 输出动作块、反归一化动作、推理耗时、峰值显存和异常统计。
- 生成离线可视化：预测轨迹与示范动作叠加，不执行机器人命令。

交付物：

- `scripts/run_inference.py`。
- `scripts/visualize_action_chunk.py`。
- `reports/inference_smoke_test.md`。
- 至少一个脱离训练代码即可重放的 golden input/output 样本。

验收标准：

- 单卡成功加载模型并完成至少 100 个真实帧的推理。
- 输出 shape 为 `[batch, 50, 7]`，且没有 NaN/Inf。
- 相同软件版本、输入、seed 下首个动作块可重复。
- 100 个样本无 shape、视频解码或归一化异常。
- 记录冷启动与热启动延迟，不把离线延迟等同于论文的机器人 command latency。

### 里程碑 A：可运行推理基线

达到阶段 3 验收标准后，视为 L1 完成。此时应先冻结一份环境、配置、golden sample 和回归测试，再进入论文算法扩展。

### 阶段 4：数据质量与离线评估基线

预计工期：2～4 个工程日。

任务：

- 扫描 2,622 episodes 的 parquet/video 完整性、帧数、时间戳和取值范围。
- 统计每个任务、subtask、接触阶段、力/力矩和动作分布。
- 使用 episode 级划分，禁止随机按 frame 划分导致时序泄漏。
- 默认建立 train/validation/test = 80/10/10；若任务泛化评估需要，另建 held-out-task split。
- 定义离线指标：
  - action MAE/RMSE；
  - 平移与旋转误差；
  - gripper accuracy/MAE；
  - action chunk smoothness；
  - wrench profile MAE、峰值误差与超限召回率；
  - 推理吞吐、延迟和显存。
- 说明行为克隆离线误差不能替代闭环成功率。

交付物：

- `scripts/audit_dataset.py`。
- `configs/splits/*.json`。
- `reports/dataset_audit.md`。
- `src/facet0/evaluation/offline_metrics.py`。

验收标准：

- 全量元数据扫描完成，无 silent skip。
- 所有 split 可由固定 seed 和 episode id 重建。
- 指标在人工构造样例上通过单元测试。

### 阶段 5：action-wrench 联合预测近似复现

预计工期：4～7 个工程日，另加训练时间。

目标：复现论文的 semantic-contact alignment 核心，即同时生成未来动作和未来 wrist-wrench profile。

任务：

- 构造目标 `Y ∈ R^(50×13)`：7 维动作 + 6 维未来 wrench。
- 使用 causal wrench history，默认 `K=10`，严格禁止未来信息进入 observation。
- 设计两个受控实现并做消融：
  1. 共享 32 维 flow head，前 13 维承载 action+wrench；
  2. 保留 7 维 action head，新增独立 6 维 wrench head。
- 按论文实现 conditional flow matching：`τ ~ Beta(1.5, 1)`。
- wrench loss 初始权重设置为 `λ_pre = 0.1`。
- 先冻结大部分 VLM，仅训练新增 head/adapter；再根据显存和收敛情况尝试 LoRA。
- 在 32～128 个短序列上执行 overfit test，再运行完整训练。

交付物：

- `src/facet0/models/joint_action_wrench.py`。
- `src/facet0/training/train_alignment.py`。
- 两种 head 设计的消融配置及报告。

验收标准：

- 小数据 overfit 时 action/wrench loss 均显著下降。
- causal leakage test 通过。
- validation wrench MAE 优于“保持当前 wrench 不变”的基线。
- 新模型仍能通过阶段 3 的动作推理回归测试。

### 阶段 6：VQA 交替训练近似复现

预计工期：3～6 个工程日，另加训练时间。

任务：

- 从数据中的 task、subtask 和阶段标签生成三类监督：当前子任务、总体任务、下一条指令。
- 实现 structured attention mask，检查 action 与文本分支的信息流。
- 使用独立 optimizer，按论文 4 次 action-wrench update 对 1 次 VQA update 交替。
- 以论文的 32K action branch、8K VQA updates 作为目标预算；先做 1% 和 10% 缩放实验。
- 对自动生成文本标签进行抽样人工审核，防止错误标签主导训练。

交付物：

- `src/facet0/data/vqa_targets.py`。
- `src/facet0/training/train_multitask.py`。
- VQA accuracy/F1 与动作指标联合报告。

验收标准：

- 更新比例与 optimizer state 经日志核验正确。
- 结构化 mask 的单元测试证明不存在 label leakage。
- VQA 学习不导致动作 validation 指标明显退化；退化阈值在实验前固定。

### 阶段 7：Action-Wrench Critic 与价值引导

预计工期：1～2 周，另加 rollout/训练时间。

此阶段无法仅靠公开离线示范严格复现。论文需要部署 rollout、失败、干预和恢复数据；实现必须区分论文明确参数与实验假设。

任务：

- 定义 distributional Action-Wrench Critic 接口和四个辅助 head。
- 实现 return distribution、regression distance、credit horizon 和 regime quantiles；这些是论文未完全指定的实验项。
- 建立 reward 配置：任务完成、效率和 wrench violation，所有系数可追溯。
- 实现 classifier-free prompt tag 和候选 action chunk ensembling/ranking。
- 先用公开数据构建 synthetic/offline proxy test，仅验证数值与排序逻辑。
- 在具备真实 rollout 后，再验证 intervention rate、recovery rate 与成功率。

交付物：

- `src/facet0/models/action_wrench_critic.py`。
- `src/facet0/training/train_critic.py`。
- `src/facet0/policies/value_guided_policy.py`。
- critic calibration、候选排序与消融报告。

验收标准：

- critic loss、分位数或分布输出数值稳定。
- 人工构造的高风险 wrench trajectory 排名低于安全 trajectory。
- candidate ranking 可被禁用，禁用时严格退化为阶段 3/6 的基线策略。
- 未采集闭环数据前，不声称复现论文 RL 性能。

### 阶段 8：FACET bottleneck 与局部 TD3 适配

预计工期：1～2 周，另加真机数据时间。

任务：

- 冻结 semantic-contact encoder，提取论文描述的四组 256 维 embedding，拼接为 1024 维表示。
- 实现 bounded absolute-action actor、twin critics、target networks 与辅助 wrench head。
- 将所有论文未给定参数配置化：学习率、折扣因子、target update、BC 权重、wrench prediction 权重和 replay 策略。
- 用离线数据验证 TD3+BC 风格训练，再用十条示范进行 few-shot protocol。
- 报告实际可训练参数比例，并与论文 6.6% 对比。

交付物：

- `src/facet0/models/bottleneck.py`。
- `src/facet0/models/local_td3.py`。
- `src/facet0/training/train_local_adaptation.py`。
- few-shot adaptation 实验报告。

验收标准：

- 冻结参数不产生梯度或 optimizer state 更新。
- twin critic、target update 和 action bounds 单元测试通过。
- 十示范实验可重复，并同时报告均值、方差和随机种子。

### 阶段 9：真机运行时与安全控制

预计工期：2～4 周，强依赖具体硬件和现场条件。

前置条件：明确机械臂型号、末端执行器、六轴力传感器、三相机标定方式和控制 API。

任务：

- 建立 perception → policy → safety filter → compliant controller 的分层运行时。
- 目标频率参考论文：策略 5～10 Hz、refinement 20 Hz、compliant controller 200 Hz。
- 实现 workspace bounds、每步平移/旋转限制、速度/加速度限制、wrench envelope、急停和 watchdog。
- 所有动作先在 shadow mode 记录，不下发；再进行无接触自由空间测试；最后逐步进入低力接触。
- 对每项真机实验记录输入、动作、wrench、安全裁剪、人工干预与视频。

交付物：

- `src/facet0/robot/` 适配器。
- 安全配置与操作检查表。
- shadow/free-space/contact 三阶段验收报告。

验收标准：

- 通信中断、推理超时和异常输出均触发安全停止。
- 安全裁剪逻辑经过仿真、回放及低速真机测试。
- 未通过安全评审前，模型输出不能绕过控制层直接发送到机器人。

### 阶段 10：论文级评估

预计工期：取决于硬件、任务夹具和 rollout 数量。

任务：

- 严格定义 RAM、CPU、Disk、GPU 和 retention-bar close 五项任务的起始状态、成功判据和最大时长。
- 每种方法使用相同初始化分布、试验次数和安全约束。
- 重现论文消融链：基础策略 → semantic-contact alignment → value-guided refinement → local adaptation。
- 报告成功率置信区间，而不仅是单一均值。
- 对无法与论文匹配的硬件、数据和超参数逐项披露。

最终验收：

- 代码、配置、checkpoint hash、数据 split、随机种子和评估日志可追溯。
- 明确区分“官方权重推理结果”“近似复现结果”和“论文原始结果”。
- 无法匹配的结果应形成差异分析，不能通过事后改变成功判据来对齐。

## 6. 测试策略

### 6.1 单元测试

- state/action padding 与切片。
- norm/unnorm 往返。
- 图像 resize、通道和范围。
- quaternion/rotation representation（确认实际数据为旋转向量后再固定）。
- 50 步 chunk 边界及 episode 尾部 mask。
- causal wrench history。
- critic target、TD3 target update 和 action clipping。

### 6.2 集成测试

- 单 episode：parquet + 三视频 → OpenPI observation。
- observation → checkpoint → 50×7 action。
- checkpoint restore 在 CPU 与单 GPU 上的一致性。
- policy server 与本地直接推理的一致性。

### 6.3 回归测试

- 固定 golden sample，保存输入 hash、配置、seed 和有限精度输出摘要。
- 依赖升级后检查参数树、首步动作、延迟和峰值显存变化。
- 允许不同硬件上的浮点微差，但阈值必须预先定义。

## 7. 实验追踪与版本管理

每次实验至少记录：

- Git commit 和未提交 diff 状态。
- OpenPI commit。
- checkpoint、norm stats 和 split 文件 SHA-256。
- 完整配置及参数来源标签。
- GPU、driver、依赖版本。
- seed、开始/结束时间、训练步数和样本数。
- train/validation 指标、峰值显存和异常退出原因。

模型命名建议：

```text
facet0-{stage}-{backbone}-{data_release}-{yyyymmdd}-{short_commit}
```

## 8. 主要风险与缓解措施

| 风险 | 影响 | 缓解措施 |
|---|---|---|
| OpenPI 版本与 checkpoint 不匹配 | 无法恢复参数 | 参数树比较；必要时定位历史 commit；保留原始权重 |
| 官方 FACET 代码发布后接口不同 | 本地实现需要迁移 | 将扩展集中在 wrapper/adapter；避免 fork 大量上游代码 |
| 当前只有约 50 小时公开数据 | 无法重现 1,000 小时训练效果 | 把结果标为 public-subset；不与论文数字直接等价 |
| checkpoint 未包含 critic/TD3 | 无法恢复完整 post-training | 将其作为独立近似实现；等待官方代码或权重 |
| 论文超参数缺失 | 结果不可唯一复现 | 配置化、消融、记录参数来源；向作者集中提问 |
| 32 GB 显存不足以全参训练 | OOM 或训练过慢 | 冻结主干、LoRA、gradient checkpointing、多卡 FSDP |
| 三相机语义映射不明确 | 推理质量显著下降 | 用数据样本可视化确认；映射可配置并做排列消融 |
| 绝对动作与坐标系不一致 | 真机危险或完全失效 | 明确 base/tool frame、单位和旋转表示；shadow mode 验证 |
| 离线指标与闭环表现弱相关 | 误判模型能力 | 离线指标只作回归；最终结论依赖统一闭环协议 |
| wrench bias、漂移或时间错位 | 接触判断失真 | 零偏校准、滤波、同步审计和传感器健康检查 |

## 9. 需要向作者确认的问题

在进入阶段 5～9 前，建议一次性向作者询问：

1. checkpoint 对应的 OpenPI commit 和完整 `Pi0Config`。
2. 三路相机到模型 image keys 的准确映射及预处理方式。
3. rotation 的表示、单位和坐标系定义。
4. flow inference 的采样步数、积分方法和 guidance 实现。
5. joint action-wrench 是共享 32 维 head 还是独立 wrench head。
6. wrench history encoder、fusion 模块和 bottleneck 的准确结构。
7. critic 的分布参数化、四个 head、regression distance 和 quantiles。
8. reward 中各项系数、phase-specific wrench limits 和接触阈值。
9. TD3 的学习率、折扣、target update、BC/prediction 权重和 replay 组成。
10. 是否会发布 critic、bottleneck、local actor、VLM judge 和 controller checkpoint。
11. ManuFacet-1K 其余约 950 小时数据的发布时间与 release 划分。

## 10. 推荐执行顺序与近期任务清单

近期只推进到里程碑 A，预计 4～9 个工程日：

1. 固定 OpenPI 和运行环境。
2. 完成 Orbax 参数树与 π0.5 配置匹配。
3. 实现 ManuFacet 三相机、13 维 state 和 7 维 action adapter。
4. 跑通一个真实 frame 的 checkpoint 推理。
5. 扩展到 100 个真实样本，生成动作轨迹和性能报告。
6. 冻结 golden sample 与自动回归测试。

里程碑 A 完成后再评审是否投入阶段 5。建议的继续条件是：

- checkpoint 可以无参数缺失地加载；
- 数据预处理不存在未解释字段；
- 输出数值稳定且动作尺度合理；
- 官方代码仍未发布，或其发布时间不足以替代本地近似实现；
- 已明确后续目标是研究性近似复现，而非宣称逐位等价于官方训练系统。

## 11. 完成定义

### L1 完成定义

- 官方 checkpoint 在本机单 GPU 可加载。
- 真实 ManuFacet 样本可端到端生成 50×7 动作块。
- 输入映射、归一化、随机性、性能和限制均有文档与测试。

### L2 完成定义

- action-wrench、VQA、critic、value-guided policy 和 local TD3 均有独立实现、配置、测试和消融。
- 所有论文未公开部分清晰标注为推测或实验选择。
- 在固定公开 split 上提供可重复的离线结果。

### L3 完成定义

- 在匹配或明确披露差异的真机平台上完成五任务闭环评估。
- 使用固定成功判据、足够试验次数和置信区间报告结果。
- 对论文结果差距给出数据、模型、控制和硬件四方面的归因分析。
