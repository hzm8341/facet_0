# FACET-0 复现项目交接文档

> 交接日期：2026-09-03  
> 项目目录：`/home/hzm/code/facet_0`  
> 当前状态：阶段 0～5 的公开数据离线工作已完成；阶段 6～10 已建立可测试的软件
> proxy/安全协议，但完整训练与真机评估受未发布标签、rollout 和硬件阻塞。

## 1. 项目目标

本项目旨在分层复现 FACET-0：

1. 加载官方 checkpoint 并完成真实数据离线推理。
2. 适配公开 ManuFacet 数据和 OpenPI π0.5 接口。
3. 近似实现论文中的 action-wrench、VQA、critic 和局部 TD3。
4. 在具备匹配硬件后，开展真机闭环评估。

当前已完成前两项的工程基线，并完成全量公开数据的基础审计。尚未开始模型训练，也没有向机器人下发动作。

详细路线见 [FACET0_REPRODUCTION_PLAN.md](./FACET0_REPRODUCTION_PLAN.md)。

## 2. 当前结论

### 2.1 Checkpoint 可以直接使用 OpenPI π0.5 加载

已验证的配置：

```python
Pi0Config(
    pi05=True,
    action_dim=32,
    action_horizon=50,
    max_token_len=200,
    paligemma_variant="gemma_2b",
    action_expert_variant="gemma_300m",
)
```

验证结果：

- checkpoint 参数叶节点：51；
- OpenPI 期望参数叶节点：51；
- missing keys：0；
- unexpected keys：0；
- shape mismatch：0；
- 参数总量：3,353,433,872；
- RTX 5090 单卡恢复成功。

权重中没有发现独立命名的 critic、TD3、FACET bottleneck、VLM judge 或 value head。因此当前发布内容应视为部署 actor checkpoint，不能用来恢复完整 RL 训练状态。

### 2.2 输入输出语义

模型输入：

- 三路 224×224 RGB 图像；
- 13 维 state：`[x,y,z,rx,ry,rz,gripper,fx,fy,fz,tx,ty,tz]`；
- 自然语言 prompt；
- state 最终补零到 OpenPI 的 32 维接口。

模型输出：

- 50 步 action chunk；
- 模型内部维度为 32；
- 仅前 7 维被解释为末端位姿与夹爪动作。

### 2.3 动作必须进行 absolute/delta 转换

公开 parquet 中的前六维 action 是绝对末端位姿，但 checkpoint action stats 以零附近的小变化量为中心。当前管线采用：

```text
训练输入：delta_action[0:6] = absolute_action[0:6] - state[0:6]
模型输出：absolute_action[0:6] = predicted_delta[0:6] + state[0:6]
夹爪维度不参与 delta 转换
```

缺少该步骤时，模型输出会被错误解释为接近零的绝对位姿。

### 2.4 公开数据与模型的夹爪方向相反

全量公开数据的 state/action 夹爪均值约为 `0.2008`，checkpoint action stats 均值为 `0.7733`。结合模型配置 `1=closed, 0=open`，当前适配层在输入时执行：

```python
model_gripper = 1.0 - public_gripper
```

该转换同时作用于 state 和训练 action。模型输出保持官方模型接口的 `1=closed` 约定。

100 样本上，使用错误方向时首步夹爪 MAE 为 0.760；修正后为 0.223。

### 2.5 相机映射仍属于推断

当前映射：

| ManuFacet 字段 | OpenPI 字段 |
|---|---|
| `view1` | `base_0_rgb` |
| `hand` | `left_wrist_0_rgb` |
| `view2` | `right_wrist_0_rgb` |

该映射依据公开字段名推断，官方 FACET 代码尚未确认。代码中保留为可配置项，不应在论文结论中称为官方映射。

## 3. 环境

### 3.1 固定版本

- GPU：NVIDIA GeForce RTX 5090，32 GB；
- Driver：595.84；
- OpenPI commit：`215abfb217dbac7d5f1273282331b9b1866c0479`；
- Python：3.11.16，由 uv 管理；
- JAX/JAXLIB：0.5.3；
- Flax：0.10.2；
- Orbax checkpoint：0.11.13；
- PyTorch：2.7.1；
- Transformers：4.53.2。

环境位置：

```text
.tools/uv
third_party/openpi/
third_party/openpi/.venv/
```

不要使用系统 Python 3.12 运行本项目。完整安装说明见 [ENVIRONMENT_SETUP.md](./ENVIRONMENT_SETUP.md)。

### 3.2 激活方式

无需激活 shell，可以直接使用固定解释器：

```bash
cd /home/hzm/code/facet_0
PYTHONPATH=src third_party/openpi/.venv/bin/python <script>
```

运行 JAX 时建议：

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false
```

## 4. 数据与模型位置

```text
Facet-0.pdf
Facet-0/facet0-post-training/
├── config.json
├── assets/norm_stats.json
└── params/                       # Orbax checkpoint

ManuFacet-1K/Facet0-1/
├── data/                         # 2,622 parquet
├── videos/                       # 7,866 mp4
└── meta/
```

公开数据实际规模：

- 2,622 episodes；
- 2,702,153 frames；
- 7,866 videos；
- 约 50.04 小时；
- GPU/RAM 两个 task family。

论文描述约 1,000 小时和三个 robot embodiment，当前公开的 `Facet0-1` 只是其中一部分。

## 5. 已完成阶段

### 阶段 0：环境

- OpenPI 已固定 commit；
- uv 环境安装完成；
- JAX GPU backend 验证成功；
- 1024×1024 GPU 矩阵计算结果有限。

### 阶段 1：Checkpoint 兼容性

- 完成 Orbax 参数 inventory；
- 与 π0.5 参数树逐叶比较；
- 完成单 GPU 全量恢复。

### 阶段 2：数据适配

- parquet signal 读取；
- OpenCV 随机视频帧读取；
- 三相机 schema 转换；
- state/action shape 检查；
- absolute/delta 动作转换；
- public/model 夹爪约定转换；
- 50 步 chunk 与尾部 valid mask。

### 阶段 3：离线推理

- 单帧端到端推理成功；
- 100 个真实样本全部成功；
- 固定 noise golden replay 验证完成。

100 样本结果：

| 指标 | 结果 |
|---|---:|
| 成功 | 100/100 |
| NaN/Inf | 0 |
| 热推理 p50 | 53.563 ms |
| 热推理 p95 | 53.822 ms |
| 端到端 p50 | 62.962 ms |
| 端到端 p95 | 67.203 ms |
| 首步平移 MAE | 0.001069 m |
| 首步旋转表示 MAE | 0.007149 |
| 首步夹爪 MAE | 0.222829 |

这些是离线 open-loop 指标，不能等价为闭环成功率。

### 阶段 4：数据审计与划分

- 全部 2,622 parquet 扫描通过；
- 总帧数与元数据完全一致；
- state/action 非有限值为 0；
- 7,866 个预期视频文件全部存在；
- 建立 episode 级、按 task family 分层的 80/10/10 split。

Split：

| Split | Episodes |
|---|---:|
| Train | 2,097 |
| Validation | 261 |
| Test | 264 |

Split SHA-256：

```text
3f7363c6ad5b4fe06498f0e08778cd3809f9a354f1f24804e90bbe5f9a09c66f
```

## 6. 代码索引

### 核心代码

- `src/facet0/data/manufacet.py`：数据、parquet、视频和 chunk 读取。
- `src/facet0/data/transforms.py`：OpenPI schema、夹爪和动作转换。
- `src/facet0/policies/facet_policy.py`：checkpoint 恢复与推理 policy 构造。

### 工具脚本

- `scripts/check_environment.py`：环境与 GPU smoke test。
- `scripts/inspect_checkpoint.py`：参数树检查和完整恢复。
- `scripts/run_inference.py`：单帧真实数据推理。
- `scripts/benchmark_inference.py`：常驻模型批量推理基准。
- `scripts/verify_golden_inference.py`：跨进程 golden 数值比较。
- `scripts/audit_dataset.py`：全量 parquet 与视频路径审计。
- `scripts/create_episode_splits.py`：生成固定 episode splits。

### 测试

- `tests/test_data_adapter.py`
- `tests/conftest.py`

当前测试状态：5 passed，Ruff 无错误。

## 7. 常用复现命令

以下命令均从项目根目录执行。

### 环境检查

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
  third_party/openpi/.venv/bin/python scripts/check_environment.py \
  --output reports/environment.json
```

### Checkpoint 兼容性

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
  third_party/openpi/.venv/bin/python scripts/inspect_checkpoint.py \
  --restore \
  --output reports/checkpoint_inventory.json
```

### 单样本推理

```bash
PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false \
  third_party/openpi/.venv/bin/python scripts/run_inference.py \
  --episode 0 \
  --frame 0 \
  --seed 0 \
  --noise-seed 314159 \
  --num-steps 10 \
  --output reports/inference_example.json
```

### 100 样本基准

```bash
PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false \
  third_party/openpi/.venv/bin/python scripts/benchmark_inference.py \
  --num-samples 100 \
  --seed 20260903 \
  --num-steps 10 \
  --output reports/inference_benchmark_100.json
```

### 静态检查与测试

```bash
third_party/openpi/.venv/bin/ruff check scripts src tests
PYTHONPATH=src third_party/openpi/.venv/bin/pytest -q tests
```

## 8. Golden replay 注意事项

即使输入和 flow noise 完全相同，当前 GPU/cuDNN 路径在不同进程间也不是逐位确定的。两次参考运行：

- 平均绝对差：约 `2.99e-4`；
- 最大平移差：小于 `5e-4 m`；
- 最大旋转表示差：约 `3.12e-3`；
- 最大夹爪差：约 `1.71e-3`。

因此不要使用 output SHA 是否完全相同作为唯一回归标准。当前固定容差：

```text
translation: 5e-4
rotation:    4e-3
gripper:     2e-3
```

验证命令：

```bash
third_party/openpi/.venv/bin/python scripts/verify_golden_inference.py \
  reports/golden_inference_a.json \
  reports/golden_inference_b.json
```

## 9. 已知问题与风险

1. 官方 FACET GitHub 仍标注 `Code coming soon`，当前实现不是官方源码。
2. 三相机映射、prompt 格式和 guidance 细节尚未获得作者确认。
3. 当前 checkpoint 不包含可识别的 critic、TD3 或 bottleneck 参数。
4. 数据 task 分布严重不平衡，task 0 和 8 占绝大多数帧。
5. `info.json` 声明 20 个 tasks，但 parquet 中实际只出现 11 个 task index。
6. 当前只检查了所有视频文件存在；没有逐帧完整解码全部约 23 GiB 视频。
7. state/action 的 `rx,ry,rz` 表示仍需作者确认，当前按原始三维旋转表示透传。
8. 公开数据的 wrench 极值较大，不能直接用于真机统一安全阈值。
9. 当前推理代码绝不能绕过安全控制层直接连接机器人。

## 10. 下一位工程师的首要任务

下一阶段为论文的 action-wrench 联合预测近似复现。建议严格按以下顺序接手。

### 10.1 先确认目标表示

构造：

```text
Y.shape = [50, 13]
Y[..., 0:7]  = action
Y[..., 7:13] = future wrench
```

Observation 使用 causal wrench history，论文默认 `K=10`。必须加入 causal leakage test，确保目标时间之后的 wrench 不进入输入。

### 10.2 先实现独立实验 head

建议首先保留已验证的官方 7 维 action policy，并增加独立 6 维 wrench prediction head，而不是直接改变官方 flow head。原因：

- 可以保留阶段 3 的推理回归基线；
- 便于确认公开数据是否足以学习未来 wrench；
- 失败时容易判断是 wrench head 问题还是 action policy 被破坏。

独立 head 通过后，再实验论文描述的共享 13 维 joint flow target。

### 10.3 最小训练实验

先使用 32～128 个短序列进行 overfit：

- 冻结 PaliGemma 主干；
- action horizon = 50；
- wrench history K = 10；
- wrench loss 初始权重 `λ_pre = 0.1`；
- flow time `τ ~ Beta(1.5, 1)`；
- 记录每个 wrench channel 的 MAE 和峰值误差。

只有小数据 overfit 成功后，才进入 train split 的完整训练。

### 10.4 建议新增文件

```text
src/facet0/models/joint_action_wrench.py
src/facet0/data/wrench_windows.py
src/facet0/training/train_alignment.py
src/facet0/evaluation/offline_metrics.py
tests/test_wrench_windows.py
tests/test_joint_action_wrench.py
configs/alignment/*.yaml
```

### 10.5 阶段 5 验收标准

- causal history 无未来泄漏；
- 小数据 action/wrench loss 均能 overfit；
- validation wrench MAE 优于保持当前 wrench 不变的基线；
- 原始 FACET action 推理仍通过 golden regression；
- 所有未由论文明确给出的选择标记为 `experimental`。

## 11. 不应立即执行的事项

- 不要修改 `Facet-0/` 中的官方 checkpoint。
- 不要原地重写 `ManuFacet-1K/` 数据。
- 不要直接对 OpenPI 执行无记录的 `git pull`。
- 不要在没有小数据 overfit 的情况下启动长时间训练。
- 不要把离线 MAE 声称为真机成功率。
- 不要在坐标系、旋转表示和安全边界未确认前连接真实机器人。

## 12. 报告索引

- `reports/environment.json`：运行环境。
- `reports/checkpoint_inventory.json`：完整参数 inventory。
- `reports/openpi_compatibility.md`：checkpoint 兼容性结论。
- `reports/inference_smoke_test.md`：单样本和 100 样本推理总结。
- `reports/inference_benchmark_100.json`：100 样本逐项结果。
- `reports/golden_inference_a.json`、`golden_inference_b.json`：跨进程 golden replay。
- `reports/dataset_audit.json`：全量数据审计结果。
- `reports/dataset_audit.md`：数据分布和风险总结。

## 13. 交接检查清单

接手后建议先执行：

```bash
cd /home/hzm/code/facet_0
third_party/openpi/.venv/bin/ruff check scripts src tests
PYTHONPATH=src third_party/openpi/.venv/bin/pytest -q tests
XLA_PYTHON_CLIENT_PREALLOCATE=false \
  third_party/openpi/.venv/bin/python scripts/check_environment.py
```

预期结果：

- Ruff：`All checks passed!`
- Pytest：`5 passed`
- 环境报告：`success: true`
- JAX backend：`gpu`

以上检查通过后，即可从阶段 5 开始开发。

## 14. 2026-09-05 阶段 5 开发续报（优先于第 10 节）

第 10 节列出的独立 wrench head、因果窗口、训练入口和离线指标已经实现。当前状态不是
“阶段 5 尚未开始”，而是“训练机制通过、验证基线尚未通过”。完整实验记录见
reports/alignment_stage5_pilots.md。

本轮完成：

- wrench 使用官方 checkpoint state q01/q99 的 7:13 通道归一化；
- 支持绝对目标与相对当前 wrench 的残差目标；
- 新增 scripts/evaluate_alignment.py，在物理单位下比较 MAE、峰值误差和常数基线；
- overfit v2/v3 均稳定下降，未出现 NaN/OOM；
- 1000 步 full v2/v3 pilot 均完成；
- 原始 FACET action backbone 全程冻结，所有新权重独立保存。

固定 8 条 validation 序列的探索性结论：

- 常数基线平均通道 MAE：0.421；
- overfit-v2 绝对目标、单样本：1.733；
- full-v2 pilot、单样本：1.983；
- overfit-v3 残差目标、8 样本均值：0.786；
- full-v3 pilot、8 样本均值：0.968。

因此阶段 5 的 validation gate 仍为失败，暂时不要启动 20,000 步训练。下一优先级是实现
确定性的 residual Huber/L1 control head，并对接触变化窗口做分层/过采样；详见续报。

新增关键路径：

    configs/alignment/overfit_v2_normalized.yaml
    configs/alignment/full_v2_normalized.yaml
    configs/alignment/overfit_v3_residual.yaml
    configs/alignment/full_v3_residual.yaml
    scripts/evaluate_alignment.py
    reports/alignment_stage5_pilots.md
    runs/alignment/overfit_v2_normalized/
    runs/alignment/full_v2_pilot/
    runs/alignment/overfit_v3_residual/
    runs/alignment/full_v3_pilot/

当前完整验证为 65 passed；原第 13 节中的 5 passed 是早期数量，已经过时。

## 15. 2026-09-05 阶段 5 最终门槛与阶段 6～10 状态

确定性 residual control 使用 2,048 个分层训练窗口，零输出严格退化为保持当前 wrench
基线。validation 上使用冻结 FACET policy 动作，并只在 validation 标定 residual scale：

- scale 候选：0、0.1、0.2、0.35、0.5、0.75、1.0；
- 选中 scale：0.2；
- validation（128 序列）：1.038 vs baseline 1.070，改善 3.00%；
- untouched test（264 episodes 各一帧）：1.0005 vs baseline 1.0170，改善 1.62%；
- test 中 4/6 单通道优于基线；force/torque 分组均值均小幅改善；
- action golden regression：通过。

这只是公开子集上的离线近似，不等价于论文真机结果。oracle action、policy action 和
zero-action 消融报告均保存在 reports/，不得只引用 oracle 结果。

阶段 6～10 新增的软件边界：

- VQA 元数据 proxy、structured mask 和 4:1 update scheduler；
- distributional critic、四个实验性辅助 head、reward proxy 与候选排序；
- 四流 1024 维 bottleneck、bounded actor、twin critic、TD3 target 与 BC loss；
- fail-closed robot safety filter、shadow-only 安全配置；
- 五任务评估协议骨架和 95% Wilson 区间。

仍无法在本机完成的外部依赖：

1. 公共数据没有作者编写的 subtask/next-instruction 文本标签；
2. 没有失败、人工干预、恢复和 reward rollout，不能训练/校准论文 critic；
3. 没有论文的 local-adaptation replay 与对应十示范 reward；
4. 没有机械臂、相机/力传感器标定、控制 API 和安全评审；
5. 因此不能产生阶段 9 真机结果或阶段 10 的五任务成功率。

在这些依赖到位前，不得将 synthetic tests 或离线 MAE 写成论文级复现。
