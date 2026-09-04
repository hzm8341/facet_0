# FACET-0 与 OpenPI π0.5 兼容性报告

> 验证日期：2026-09-03  
> OpenPI commit：`215abfb217dbac7d5f1273282331b9b1866c0479`  
> checkpoint：`Facet-0/facet0-post-training/params`

## 结论

FACET-0 发布的 Orbax checkpoint 与当前固定版本的标准 OpenPI π0.5 模型结构完全兼容。

验证使用的模型配置为：

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

参数比对及 GPU 恢复均已通过，不需要重命名、裁剪或补充参数。

## 参数树比对结果

| 检查项 | 结果 |
|---|---:|
| checkpoint 参数叶节点 | 51 |
| OpenPI 期望参数叶节点 | 51 |
| 缺失 key | 0 |
| 多余 key | 0 |
| shape mismatch | 0 |
| 总参数量 | 3,353,433,872 |
| GPU 完整恢复 | 成功 |
| GPU 恢复并装载耗时 | 约 8.9 秒 |

顶层参数量：

| 模块 | 参数量 |
|---|---:|
| PaliGemma（含 action expert） | 3,351,268,080 |
| action input projection | 33,792 |
| action output projection | 32,800 |
| time MLP input | 1,049,600 |
| time MLP output | 1,049,600 |

完整逐叶 inventory 位于 `reports/checkpoint_inventory.json`。

## 已确认的结构信息

- 模型是 OpenPI `pi05`，不是 π0 或 π0-FAST。
- vision encoder 是 SigLIP So400m/14 对应结构，输入模型分辨率为 224×224。
- PaliGemma 主干宽度为 2048，action expert 宽度为 1024。
- action head 的全局输入/输出维度均为 32。
- FACET 发布的 normalization stats 表明：
  - state 使用前 13 维；
  - action 使用前 7 维；
  - 其余维度为 OpenPI padding。
- π0.5 配置启用 `discrete_state_input=True`，因此状态会由 OpenPI transform 编码进离散输入，而不是使用 π0 的连续 `state_proj`。

## 未包含的独立模块

对全部 51 个参数路径进行名称与结构检查，没有发现独立的：

- Action-Wrench Critic；
- TD3 actor 或 twin critics；
- FACET bottleneck；
- VLM judge；
- 独立 value head。

参数路径末尾的 `value` 是 Flax NNX checkpoint 的数组包装节点，不是论文所述价值网络。

这说明当前 checkpoint 可以作为最终 action policy 使用，但不能从中直接恢复论文的 critic、VLM judge 或局部 TD3 训练状态。它们可能没有发布，或已通过策略蒸馏/微调折叠进最终 actor；仅凭参数树无法区分这两种情况。

## 仍需通过数据适配确认的内容

模型结构兼容不等于推理语义已经兼容。进入端到端推理前仍需确认：

1. `view1`、`hand`、`view2` 到三个 OpenPI image key 的准确映射。
2. 三路图像的裁剪方式；当前只能确定模型最终输入为 224×224。
3. 13 维状态写入 π0.5 discrete-state tokenizer 的格式。
4. 数据中的 `rx, ry, rz` 是旋转向量、欧拉角还是其他表示，以及角度单位。
5. prompt 来自 episode task、subtask 文本还是两者组合。
6. FACET 配置中的 guidance scale 如何接入标准 OpenPI sampling。

这些问题属于阶段 2 的数据与推理适配，不影响本报告关于参数结构兼容的结论。

## 重跑方法

```bash
cd /home/hzm/code/facet_0
XLA_PYTHON_CLIENT_PREALLOCATE=false \
  third_party/openpi/.venv/bin/python scripts/inspect_checkpoint.py \
  --restore \
  --output reports/checkpoint_inventory.json
```

命令返回码为 0 且 JSON 中 `summary.compatible`、`restore_test.success` 均为 `true` 时，兼容性验证通过。

