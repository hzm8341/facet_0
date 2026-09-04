# FACET-0 首次离线推理报告

> 日期：2026-09-03  
> 状态：阶段 3 已通过，包括单样本 smoke test 和 100 样本常驻进程基准。

## 结果

官方 FACET-0 checkpoint 已在公开 ManuFacet episode 0、frame 0 上完成端到端推理：

- checkpoint 恢复：成功；
- 三路视频解码：成功；
- 图像预处理：480×640 RGB → 224×224 RGB；
- π0.5 prompt/state tokenization：成功；
- flow sampling steps：10；
- 输出 shape：50×7；
- NaN/Inf：无；
- checkpoint 恢复耗时：约 4.8 秒；
- 首次进程内推理及 JIT：约 10.0 秒。

完整数值结果位于 `reports/inference_episode000000_frame0000.json`。首次 JIT 时间不能与论文报告的 50 ms command latency 直接比较；需要在同一常驻进程内完成 warm-up 后再测量稳态延迟。

## 动作语义修正

公开 parquet 的 `action` 是 7 维绝对末端目标，frame 0 的前六维与当前 state 接近。但 checkpoint 的 action normalization stats 显示：

- 前六维均以 0 附近的小变化量为中心；
- 第七维夹爪保持约 `[0, 1]` 的绝对值；
- 第 8～32 维是 padding。

因此推理管线采用：

1. 训练输入：前六维 absolute action 减当前 state，转为 delta；夹爪不变。
2. 模型输出：反归一化后将前六维 delta 加回当前 state；夹爪不变。
3. 只返回前七维作为 FACET 动作。

加入该变换后，首个预测的六维位姿与当前绝对位姿处于相同尺度。没有这一步时，模型输出会被错误解释为接近零的绝对位姿。

## 视频同步检查

episode 0 的 parquet 长度为 1,905。三路视频的容器元数据与完整解码帧数均为 1,905，帧率均为 15 FPS：

| Camera | Metadata frames | Decoded frames | FPS |
|---|---:|---:|---:|
| view1 | 1,905 | 1,905 | 15 |
| hand | 1,905 | 1,905 | 15 |
| view2 | 1,905 | 1,905 | 15 |

该样本没有发现相机间帧数错位。全量一致性仍将在阶段 4 数据审计中完成。

## 当前限制

- 相机映射 `view1 → base`、`hand → left wrist`、`view2 → right wrist` 是根据公开字段名作出的推断，尚无作者代码确认。
- 当前使用 `tasks.jsonl` 中当前 frame 的 `task_index` 生成 prompt；仍需确认官方是否使用更短的 subtask prompt。
- 第一个样本的预测夹爪值与示范动作差异较大，单样本不能判断是随机采样、相机/prompt 映射还是策略误差。
- 当前只验证了一个真实帧，不能据此声称复现论文精度或成功率。
- checkpoint 没有提供 guidance 的专用实现；当前使用标准 OpenPI π0.5 sampler，相当于基础 guidance 设置。

## 100 样本基准

固定 seed `20260903`，从不同 episode 中无放回抽取 100 个 episode，再在每个 episode 内均匀抽取一个 frame。每个样本使用可追溯的显式 flow noise。

| 指标 | 结果 |
|---|---:|
| 成功推理 | 100 / 100 |
| NaN/Inf | 0 |
| 热推理 p50 | 53.563 ms |
| 热推理 p95 | 53.822 ms |
| 端到端 p50 | 62.962 ms |
| 端到端 p95 | 67.203 ms |
| 首步平移 MAE | 0.001069 m |
| 首步旋转表示 MAE | 0.007149 |
| 首步夹爪 MAE | 0.222829 |

详细记录、每个输入/noise/output 的 SHA-256、逐通道 chunk MAE 和样本分布位于 `reports/inference_benchmark_100.json`。

公开数据中的夹爪字段表示连续开度，大值更开；模型配置规定输出大值更闭。将 public state/action 的第七维转换为 `1-x` 后，100 样本首步夹爪 MAE 从错误解释下的 0.760 降至 0.223。该转换已固定在 FACET input adapter 中。

## Golden replay

episode 0、frame 0 使用固定 noise seed `314159` 在两个独立进程中重放。noise SHA-256 完全相同，输出均为有限 float32，但 GPU/cuDNN 计算不是逐位确定的：平均绝对差为 `2.99e-4`，最大差出现在旋转通道，为 `3.12e-3`。

因此 golden regression 不使用输出文件 SHA 逐位相等，而使用预先固定的逐通道绝对容差：平移 `5e-4 m`、旋转表示 `4e-3`、夹爪 `2e-3`。本次跨进程重放通过该标准。验证入口为 `scripts/verify_golden_inference.py`。

## 下一步验收

1. 对三相机的 6 种排列和短/长 prompt 做小规模消融，避免把未经确认的映射固化为结论。
2. 在阶段 4 中扩大 task-index 覆盖并执行全量数据审计。
3. 在获得官方代码后核对相机、prompt、夹爪和 guidance 语义。
