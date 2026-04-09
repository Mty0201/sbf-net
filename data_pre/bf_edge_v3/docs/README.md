# BF Edge v3

`bf_edge_v3` 是一套面向建筑立面语义点云的边缘监督预处理流程。

当前实现已经覆盖：

- 四阶段单场景流程
- pointwise edge supervision 构建
- 一步式数据集级批处理（`rebuild_edge_dataset_inplace.py`）
- C++ 加速后端（`bf_edge_cpp`，约 24x 加速）
- 紧凑训练监督 `edge.npy`

项目目录结构：

- `scripts/`
  入口脚本。包含单场景四阶段脚本和一步式数据集批处理脚本。
- `core/`
  核心算法实现。boundary centers、local clusters、supports、pointwise 四个阶段的核心逻辑。
- `cpp/`
  C++ 加速实现。通过 pybind11 暴露给 Python，批处理脚本自动检测并使用。
- `utils/`
  公共工具。共享的 IO、导出和基础工具函数。
- `tests/`
  测试。config、contract、equivalence、validation、density-adaptive 等测试。
- `docs/`
  使用说明文档。

建议阅读顺序：

1. 先看 [PIPELINE.md](PIPELINE.md)
   了解单场景四阶段流程和批处理流程。
2. 再看 [DATASET_FORMAT.md](DATASET_FORMAT.md)
   了解数据集的目录格式。

快速开始（一步式批处理，推荐）：

```bash
# 如果还没编译 C++ 加速：
cd data_pre/bf_edge_v3/cpp && pip install --no-build-isolation .

# 批处理整个数据集（4 进程并行）：
cd data_pre/bf_edge_v3
python scripts/rebuild_edge_dataset_inplace.py \
    --input /path/to/dataset --workers 4
```

启动时会打印 `Backend: C++` 或 `Backend: Python` 表示当前使用的后端。
