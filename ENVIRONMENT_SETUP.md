# FACET-0 开发环境

本项目使用官方 OpenPI 的锁定环境。原始 checkpoint 和 ManuFacet 数据保持只读。

## 固定版本

- OpenPI commit：`215abfb217dbac7d5f1273282331b9b1866c0479`
- OpenPI commit date：`2026-08-25T00:27:12+08:00`
- uv：项目内 `.tools/uv`
- Python：由 uv 管理的 CPython 3.11.16
- 依赖锁：`third_party/openpi/uv.lock`

OpenPI 的两个 submodule 固定为：

- `third_party/aloha`: `d1dc83afd89ded4379851257fe5d85632d31d5ec`
- `third_party/libero`: `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`

## 重建环境

```bash
cd /home/hzm/code/facet_0
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$PWD/.tools" sh
GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules \
  https://github.com/Physical-Intelligence/openpi.git third_party/openpi
git -C third_party/openpi checkout 215abfb217dbac7d5f1273282331b9b1866c0479
git -C third_party/openpi submodule update --init --recursive
cd third_party/openpi
GIT_LFS_SKIP_SMUDGE=1 ../../.tools/uv sync --frozen
```

如果目录已存在，只需运行最后两行进行一致性同步。

## 环境验证

必须使用 OpenPI 虚拟环境中的 Python：

```bash
cd /home/hzm/code/facet_0
XLA_PYTHON_CLIENT_PREALLOCATE=false \
  third_party/openpi/.venv/bin/python scripts/check_environment.py \
  --output reports/environment.json
```

检查成功时，报告中的以下字段应满足：

- `success: true`
- `jax_smoke_test.backend: "gpu"`
- `jax_smoke_test.all_finite: true`
- `openpi.commit` 与上述固定 commit 相同
- `assets.checkpoint_exists` 和 `assets.dataset_exists` 均为 `true`

## 更新规则

不要直接对 OpenPI 执行无记录的 `git pull`。若后续需要升级：

1. 新建兼容性测试分支或工作副本。
2. 记录旧、新 commit。
3. 重跑 checkpoint 参数树比较与 golden inference。
4. 只有全部回归通过后，才更新本文档中的固定版本。

