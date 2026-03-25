# BF Edge v3

`bf_edge_v3` 是一套面向建筑立面语义点云的边缘监督预处理流程。

当前实现已经覆盖：

- 四阶段单场景流程
- pointwise edge supervision 构建
- 两步式数据集级批处理
- 紧凑训练监督 `edge.npy`

项目目录结构：

- `scripts/`
  入口脚本。这里放单场景四阶段脚本和两步式数据集批处理脚本。
- `core/`
  核心算法实现。这里放 boundary centers、local clusters、supports、pointwise 四个阶段的核心逻辑。
- `utils/`
  公共工具。这里放共享的 IO、导出和基础工具函数。
- `docs/`
  使用说明文档。

建议阅读顺序：

1. 先看 [PIPELINE.md](/home/mty0201/Pointcept/pointcept/datasets/preprocessing/bf_edge_v3/docs/PIPELINE.md)
   了解单场景四阶段流程和两步式批处理流程。
2. 再看 [DATASET_FORMAT.md](/home/mty0201/Pointcept/pointcept/datasets/preprocessing/bf_edge_v3/docs/DATASET_FORMAT.md)
   了解原始数据集、补 supports 后数据集和 edge 数据集的目录格式。

运行方式：

- 建议在仓库根目录下直接运行 `scripts/` 下的入口脚本，例如：

```bash
python pointcept/datasets/preprocessing/bf_edge_v3/scripts/build_boundary_centers.py --help
```

当前 `scripts/` 入口会在运行时定位 `bf_edge_v3` 根目录，因此不需要额外修改 `PYTHONPATH`。
