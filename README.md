# FACET-0 公开资产复现与离线研究实现

本仓库用于验证公开 FACET-0 checkpoint、适配 ManuFacet-1K 数据，并在官方训练代码和
完整 rollout 尚未公开的条件下，实现可测试、可追溯的 action-wrench、VQA、critic、
局部 TD3 和安全控制近似模块。

## 当前结论

当前已经完成：

- 官方 FACET-0 checkpoint 在 OpenPI pi0.5 上无缺失恢复；
- 真实 ManuFacet 帧到 50 x 7 action chunk 的端到端离线推理；
- 100 个真实样本推理全部成功，无 NaN/Inf；
- checkpoint 参数树与标准 pi0.5 配置 51 个叶节点逐项匹配；
- 因果 wrench history、未来 wrench target、归一化和泄漏测试；
- 独立 stochastic flow wrench head 及确定性 residual control head；
- validation/test wrench baseline 对比；
- VQA mask/调度、distributional critic、value ranking、bottleneck 和 TD3+BC proxy；
- fail-closed shadow-mode 机器人安全过滤器；
- 五任务评估协议和 95% Wilson 置信区间；
- 65 项自动化测试、全仓 Ruff 和 JAX GPU smoke test。

当前没有复现论文报告的约 82% 真机成功率。该指标需要匹配的机器人、夹具、传感器
标定、控制器、约 1,000 小时训练数据以及失败/干预/恢复 rollout。公开资产只覆盖约
50 小时数据和部署 actor checkpoint。仓库中的离线 MAE、synthetic critic tests 和
shadow safety tests 不能解释为真机成功率。

详细完成边界见
[reports/phases5_10_completion_status.md](reports/phases5_10_completion_status.md)。

## 已验证配置

官方 checkpoint 与以下 OpenPI 配置匹配：

    Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=50,
        max_token_len=200,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
    )

本机验证环境：

- Python 3.11.16；
- OpenPI commit 215abfb217dbac7d5f1273282331b9b1866c0479；
- JAX/JAXLIB 0.5.3；
- Flax 0.10.2；
- NVIDIA RTX 5090 32 GB；
- 单 GPU CUDA backend。

完整版本和重建方法见 [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md)。

## 仓库结构

    .
    ├── configs/
    │   ├── alignment/          # flow 与 deterministic wrench 实验
    │   ├── evaluation/         # 论文级任务协议骨架
    │   ├── robot/              # shadow-only 安全配置
    │   └── splits/             # episode 级 train/validation/test 划分
    ├── scripts/
    │   ├── check_environment.py
    │   ├── inspect_checkpoint.py
    │   ├── run_inference.py
    │   ├── benchmark_inference.py
    │   ├── audit_dataset.py
    │   ├── evaluate_alignment.py
    │   └── evaluate_deterministic_wrench.py
    ├── src/facet0/
    │   ├── data/               # 数据读取、转换、wrench/VQA targets
    │   ├── models/             # wrench、critic、bottleneck、TD3 模型
    │   ├── policies/           # FACET policy 与 value ranking
    │   ├── training/           # alignment、VQA schedule、critic、TD3
    │   ├── evaluation/         # 离线指标与 trial protocol
    │   └── robot/              # 硬件无关安全过滤器
    ├── tests/
    ├── reports/
    ├── FACET0_REPRODUCTION_PLAN.md
    └── HANDOFF.md

以下大型或第三方资产由 .gitignore 排除，不会进入代码提交：

    Facet-0/
    ManuFacet-1K/
    Facet-0.pdf
    third_party/
    runs/

## 准备公开资产

期望目录：

    Facet-0/facet0-post-training/
    ├── assets/norm_stats.json
    ├── config.json
    └── params/

    ManuFacet-1K/Facet0-1/
    ├── data/
    ├── meta/
    └── videos/

可使用 Hugging Face CLI 下载。数据和 checkpoint 合计体积较大，执行前确认磁盘空间：

    hf download Pinelab/Facet-0 +      --local-dir Facet-0

    hf download Pinelab/ManuFacet-1K +      --repo-type dataset +      --local-dir ManuFacet-1K

请遵守原始仓库的许可证和使用条款。本仓库不重新分发 checkpoint、数据或论文 PDF。

## 环境安装

    curl -LsSf https://astral.sh/uv/install.sh +      | env UV_INSTALL_DIR="$PWD/.tools" sh

    GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules +      https://github.com/Physical-Intelligence/openpi.git third_party/openpi

    git -C third_party/openpi checkout +      215abfb217dbac7d5f1273282331b9b1866c0479

    git -C third_party/openpi submodule update --init --recursive

    cd third_party/openpi
    GIT_LFS_SKIP_SMUDGE=1 ../../.tools/uv sync --frozen
    cd ../..

所有项目命令应使用固定环境，不要使用系统 Python：

    PYTHONPATH=src third_party/openpi/.venv/bin/python ...

JAX 命令建议设置：

    XLA_PYTHON_CLIENT_PREALLOCATE=false

## 快速验证

### 1. 环境和 GPU

    XLA_PYTHON_CLIENT_PREALLOCATE=false +      third_party/openpi/.venv/bin/python scripts/check_environment.py +      --output reports/environment.json

成功报告应包含：

- success 为 true；
- jax_smoke_test.backend 为 gpu；
- jax_smoke_test.all_finite 为 true；
- checkpoint_exists 和 dataset_exists 为 true。

### 2. Checkpoint 参数兼容性

    XLA_PYTHON_CLIENT_PREALLOCATE=false +      third_party/openpi/.venv/bin/python scripts/inspect_checkpoint.py +      --restore +      --output reports/checkpoint_inventory.json

已验证结果：

- 参数叶节点 51；
- missing keys 0；
- unexpected keys 0；
- shape mismatch 0；
- 参数总量 3,353,433,872。

### 3. 单帧离线推理

    PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false +      third_party/openpi/.venv/bin/python scripts/run_inference.py +      --episode 0 +      --frame 0 +      --seed 0 +      --noise-seed 314159 +      --num-steps 10 +      --output reports/inference_example.json

成功输出 shape 为 50 x 7，且全部为有限值。

### 4. 完整自动化测试

    third_party/openpi/.venv/bin/python -m ruff check scripts src tests

    PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false +      third_party/openpi/.venv/bin/python -m pytest -q tests

当前期望：

- Ruff: All checks passed；
- Pytest: 65 passed。

固定动作回归：

    third_party/openpi/.venv/bin/python scripts/verify_golden_inference.py +      reports/golden_inference_a.json +      reports/golden_inference_b.json

GPU 算子在不同进程间可能存在轻微数值差异，因此使用预先声明的逐通道容差，而不是要求
输出文件 SHA 完全一致。

## 数据和动作语义

公开状态为 13 维：

    [x, y, z, rx, ry, rz, gripper, fx, fy, fz, tx, ty, tz]

公开动作为 7 维绝对末端目标。模型内部接口为 32 维，实际只解释前 7 维。

当前适配管线执行：

    delta_action[0:6] = absolute_action[0:6] - state[0:6]

推理后执行逆转换：

    absolute_action[0:6] = predicted_delta[0:6] + state[0:6]

公开数据的夹爪值越大越开，checkpoint 接口使用越大越闭，因此输入转换为：

    model_gripper = 1.0 - public_gripper

当前相机映射属于根据公开字段名做出的推断：

| ManuFacet | OpenPI |
|---|---|
| view1 | base_0_rgb |
| hand | left_wrist_0_rgb |
| view2 | right_wrist_0_rgb |

映射仍需 FACET 作者确认，不应描述为官方语义。

## Action-wrench 实验

### Stochastic flow head

归一化 overfit：

    PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false +      third_party/openpi/.venv/bin/python +      -m facet0.training.train_alignment +      --config configs/alignment/overfit_v3_residual.yaml

评估：

    PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false +      third_party/openpi/.venv/bin/python scripts/evaluate_alignment.py +      --config configs/alignment/overfit_v3_residual.yaml +      --head runs/alignment/overfit_v3_residual/wrench_head_final.npz +      --split validation +      --num-sequences 8 +      --num-samples 8 +      --output reports/alignment_overfit_v3_validation_ensemble8.json

flow head 在归一化后训练稳定，但没有超过“保持当前 wrench”基线，因此没有继续执行
20,000 步长训练。

### Deterministic residual control

小样本 overfit：

    PYTHONPATH=src third_party/openpi/.venv/bin/python +      -m facet0.training.train_deterministic_wrench +      --config configs/alignment/deterministic_overfit.yaml

2,048 序列分层 pilot：

    PYTHONPATH=src third_party/openpi/.venv/bin/python +      -m facet0.training.train_deterministic_wrench +      --config configs/alignment/deterministic_full_pilot.yaml

该模型为零初始化；输出零 residual 时严格等价于保持当前 wrench。训练池从 4,096 个候选
窗口中选择 2,048 个，其中 50% 为高变化窗口。

validation 上比较固定候选 scale：

    PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false +      third_party/openpi/.venv/bin/python +      scripts/evaluate_deterministic_wrench.py +      --config configs/alignment/deterministic_full_pilot.yaml +      --head runs/alignment/deterministic_full_pilot/wrench_head_final.npz +      --split validation +      --num-sequences 128 +      --action-source policy +      --residual-scales 0,0.1,0.2,0.35,0.5,0.75,1.0 +      --output reports/deterministic_validation.json

validation 选择 scale 0.2 后，应冻结配置，再进行一次 test：

    PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false +      third_party/openpi/.venv/bin/python +      scripts/evaluate_deterministic_wrench.py +      --config configs/alignment/deterministic_full_pilot.yaml +      --head runs/alignment/deterministic_full_pilot/wrench_head_final.npz +      --split test +      --num-sequences 264 +      --action-source policy +      --residual-scales 0.2 +      --output reports/deterministic_test.json

已记录结果：

| Split | Model MAE | Constant baseline | 改善 |
|---|---:|---:|---:|
| Validation，128 序列 | 1.0379 | 1.0700 | 3.00% |
| Test，264 episodes | 1.0005 | 1.0170 | 1.62% |

该提升很小，只说明公开子集上存在可学习的离线信号，不代表闭环成功率改善。

## 阶段 6–10 软件边界

### VQA

已实现 metadata proxy targets、structured attention mask 和论文描述的 4:1 更新调度。
公开数据只有数值 subtask ID，没有作者编写的语义 subtask/next-instruction 标签，因此
无法忠实训练和评估 VQA。

### Critic 和 value guidance

已实现 categorical return head、四个实验性辅助 head、two-hot target、reward proxy 和
风险感知候选排序。由于没有失败、干预、恢复和 reward rollout，只能进行数值与 synthetic
排序测试，不能声称 critic 已校准。

### Bottleneck 和 local TD3

已实现四路 256 维投影组成的 1,024 维 bottleneck、bounded actor、twin critics、Polyak
更新、TD3 bootstrap target 和 TD3+BC loss。缺少 task-local reward replay 和论文对应的
十示范数据，无法完成论文训练结果。

### 机器人安全

src/facet0/robot/safety.py 支持：

- 非有限输入立即停止；
- watchdog timeout；
- wrench envelope；
- workspace bounds；
- 单步平移/旋转限制；
- gripper clipping。

configs/robot/safety_shadow.yaml 仅用于回放和 shadow mode，其数值不是经过具体机器人
认证的安全限制。缺少硬件标定和安全评审时，不得发送真实控制命令。

### 论文级评估

configs/evaluation/robot_tasks.yaml 定义 RAM、CPU、Disk、GPU 和 retention-bar-close 五项
任务的协议骨架和必要日志字段。没有真机 trials 时不会生成或推测论文成功率。

## 主要报告

- [环境报告](reports/environment_final.json)
- [Checkpoint 兼容性](reports/openpi_compatibility.md)
- [数据审计](reports/dataset_audit.md)
- [推理报告](reports/inference_smoke_test.md)
- [Action-wrench pilots](reports/alignment_stage5_pilots.md)
- [阶段 5–10 完成边界](reports/phases5_10_completion_status.md)
- [最终 test 指标](reports/deterministic_full_pilot_test_264_policy_scale02.json)
- [开发交接](HANDOFF.md)

## 可复现性规则

- 不修改 Facet-0/ 或 ManuFacet-1K/；
- 不在原 checkpoint 目录保存训练权重；
- 每个实验使用独立 runs/ 子目录；
- 固定 OpenPI commit、数据 split、随机种子和配置；
- 未由论文明确给出的结构和超参数标记为 experimental；
- validation 用于模型/scale 选择，test 只用于冻结配置后的最终报告；
- 离线误差不得描述为真机成功率；
- OpenPI 升级前必须重跑 checkpoint inventory、golden inference 和完整测试。

## 已知限制

1. 官方训练代码、critic、TD3、bottleneck 和 VLM judge 权重未包含在公开 checkpoint 中。
2. 公开数据约 50 小时，远小于论文描述的约 1,000 小时。
3. 当前三相机语义和旋转表示仍需作者确认。
4. 视频文件已完成存在性检查，但没有逐帧完整解码约 23 GiB 视频。
5. deterministic wrench 改善幅度较小，且仍是开放环离线评估。
6. 没有机器人、传感器标定、控制 API、夹具和闭环 trial。

## 文档索引

- [环境重建](ENVIRONMENT_SETUP.md)
- [详细开发计划](FACET0_REPRODUCTION_PLAN.md)
- [交接文档](HANDOFF.md)
- [阶段 5–10 状态](reports/phases5_10_completion_status.md)
