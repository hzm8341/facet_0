# ManuFacet Facet0-1 数据审计报告

> 日期：2026-09-03  
> 范围：全部 parquet 数值与全部预期视频路径；视频码流采用抽样解码。

## 结论

公开 Facet0-1 release 的基础结构检查通过：

| 项目 | 实际 | 元数据期望 |
|---|---:|---:|
| Episodes | 2,622 | 2,622 |
| Frames | 2,702,153 | 2,702,153 |
| Videos | 7,866 | 7,866 |
| Video payload | 24,700,132,595 bytes | — |
| state 非有限值 | 0 | 0 |
| action 非有限值 | 0 | 0 |
| schema/长度/文件错误 | 0 | 0 |

完整机器可读结果位于 `reports/dataset_audit.json`。

## 数值范围

- 状态前六维范围与动作前六维范围基本一致，均是绝对末端位姿。
- state 和 action 第七维范围均为 `[0, 1]`，原始全量均值分别为 `0.200767` 和 `0.200768`。
- 取 `1-x` 后夹爪均值为约 `0.7992`，与 checkpoint action stats 的 `0.7733` 接近，进一步支持“公开数据记录开度、模型接口记录闭合度”的转换判断。
- 六轴 wrench 的观测极值较大：力分量全局范围约为 `[-85.79, 153.12]`，力矩分量约为 `[-17.01, 12.90]`。训练和真机使用前必须按 phase-specific envelope 处理，不能只用一个全局阈值。

## 任务分布

虽然 `info.json` 声明 `total_tasks = 20`，实际帧中只出现 11 个 task index。数据高度集中于：

- task 8：1,346,644 帧；
- task 0：1,220,496 帧；
- 其余 9 个 task index 合计占比较小。

subtask 分布：

| Subtask | Frames |
|---:|---:|
| 0 | 12 |
| 1 | 264,261 |
| 2 | 944,618 |
| 3 | 977,804 |
| 4 | 512,007 |
| 5 | 3,451 |

因此后续训练不能仅报告总体 frame-weighted loss，还必须按 task/subtask 分组报告；否则 task 0 和 8 会掩盖少数任务表现。

## 数据划分

已生成 episode 级、按 `task_family` 分层的固定 80/10/10 split：

| Split | Episodes |
|---|---:|
| Train | 2,097 |
| Validation | 261 |
| Test | 264 |

- Seed：`20260903`
- Split SHA-256：`3f7363c6ad5b4fe06498f0e08778cd3809f9a354f1f24804e90bbe5f9a09c66f`
- 文件：`configs/splits/facet0_1_seed20260903.json`

该划分以 episode 为最小单位，不会把同一轨迹的相邻帧泄漏到不同 split。

## 视频验证范围

- 已验证全部 7,866 个预期路径存在且文件大小可读取。
- 100 样本推理基准成功随机访问并解码了 300 个视频帧。
- episode 0 的三路视频均完成全量解码，三者与 parquet 同为 1,905 帧、15 FPS。
- 当前没有逐帧解码全部约 23 GiB 视频；若用于发布数据质量声明，应另运行长时完整码流验证。

## 后续建议

1. 训练 sampler 对少数 task/subtask 使用 episode-balanced 或 group-balanced 权重。
2. 将 task 0/8 的总体指标与 macro-averaged task 指标同时报告。
3. 在 action-wrench 训练前计算 causal wrench history 的 robust quantiles，并审查极端峰值对应的视频与接触阶段。
4. 不将当前随机 episode split 用作“未见任务泛化”结论；该结论需要单独的 held-out-task split。

