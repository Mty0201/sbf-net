# BF Edge v3 Dataset Format

## 1. 原始数据集结构

原始数据集至少按 `training / validation` 两级目录组织，每个样本目录至少包含：

- `coord.npy`
- `segment.npy`

常见可选文件：

- `color.npy`
- `normal.npy`

示例：

```text
dataset
├─ training
│  └─ 020101
│     ├─ coord.npy
│     ├─ color.npy
│     ├─ normal.npy
│     └─ segment.npy
└─ validation
   └─ 010101
      ├─ coord.npy
      ├─ color.npy
      ├─ normal.npy
      └─ segment.npy
```


## 2. 第一步补 supports 后的数据结构

运行 `build_support_dataset_v3.py` 后，原始样本目录原地补：

- `supports.npz`
- `support_geometry.xyz`

这一步结束后，不再保留 boundary centers、local clusters 等中间产物。

示例：

```text
dataset
├─ training
│  └─ 020101
│     ├─ coord.npy
│     ├─ color.npy
│     ├─ normal.npy
│     ├─ segment.npy
│     ├─ supports.npz
│     └─ support_geometry.xyz
└─ validation
   └─ 010101
      ├─ coord.npy
      ├─ color.npy
      ├─ normal.npy
      ├─ segment.npy
      ├─ supports.npz
      └─ support_geometry.xyz
```

说明：
- `supports.npz` 是后续 pointwise supervision 的必要输入。
- `support_geometry.xyz` 是快速人工验收用的可视化文件。


## 3. 一步式原地重建（推荐）

运行 `rebuild_edge_dataset_inplace.py` 后，在原始样本目录原地生成/覆盖三个文件：

- `edge.npy`
- `edge_supervision.xyz`
- `support_geometry.xyz`

不保留中间产物（boundary_centers.npz、local_clusters.npz、supports.npz）。

示例：

```text
dataset
├─ training
│  └─ 020101
│     ├─ coord.npy
│     ├─ color.npy
│     ├─ normal.npy
│     ├─ segment.npy
│     ├─ edge.npy                  ← 生成
│     ├─ edge_supervision.xyz      ← 生成
│     └─ support_geometry.xyz      ← 生成
└─ validation
   └─ 010101
      ├─ coord.npy
      ├─ color.npy
      ├─ normal.npy
      ├─ segment.npy
      ├─ edge.npy
      ├─ edge_supervision.xyz
      └─ support_geometry.xyz
```

用法：

```bash
cd data_pre/bf_edge_v3
python scripts/rebuild_edge_dataset_inplace.py \
    --input /path/to/dataset --workers 4
```


## 4. 两步式 edge 数据集结构（旧流程）

### 4.1 第一步补 supports

同上文第 2 节。运行 `build_support_dataset_v3.py` 原地补 `supports.npz` + `support_geometry.xyz`。

### 4.2 第二步生成 edge 数据集

运行 `build_edge_dataset_v3.py` 后，会生成一个新的紧凑 edge 数据集。

新数据集保留：

- 原始基础文件：
  - `coord.npy`
  - `color.npy`
  - `normal.npy`
  - `segment.npy`
- 新监督文件：
  - `edge.npy`
  - `edge_supervision.xyz`

示例：

```text
dataset_edge
├─ training
│  └─ 020101
│     ├─ coord.npy
│     ├─ color.npy
│     ├─ normal.npy
│     ├─ segment.npy
│     ├─ edge.npy
│     └─ edge_supervision.xyz
└─ validation
   └─ 010101
      ├─ coord.npy
      ├─ color.npy
      ├─ normal.npy
      ├─ segment.npy
      ├─ edge.npy
      └─ edge_supervision.xyz
```


## 5. `edge.npy` 定义

`edge.npy` 是最终训练友好的紧凑监督。

- shape：`(N, 5)`
- dtype：`float32`

列顺序固定为：

1. `vec_x`
2. `vec_y`
3. `vec_z`
4. `edge_support`
5. `edge_valid`

等价写法：

```text
edge[:, 0:3] = edge_vec
edge[:, 3] = edge_support
edge[:, 4] = edge_valid
```

说明：
- `edge_vec` 是主几何监督，表示点到最近边界支撑元投影点的位移。
- `edge_support` 是连续吸附权重，越靠近边界支撑元越高，超出监督半径后为 0。
- `edge_valid` 只是数值有效域，表示当前点是否存在可定义且位于监督半径内的最近支撑元。

兼容说明：
- 逐点导出仍会额外保留 `edge_strength.npy` 和 `edge_mask.npy`，它们分别等价于 `edge_support.npy` 和 `edge_valid.npy`。

当前 `edge.npy` 不包含 `edge_dist`、`edge_dir`、`edge_support_id`。
这些字段仅在逐阶段运行时单独输出，一步式批处理不生成。


## 6. `edge_supervision.xyz` 作用

`edge_supervision.xyz` 是 pointwise 监督的快速人工验收文件。

特点：

- 只保留 `edge_valid == 1` 的点
- 用于 CloudCompare 等工具快速查看边界带
- 不作为最终训练读取主文件，训练侧主文件是 `edge.npy`

当前字段顺序为：

1. `x`
2. `y`
3. `z`
4. `edge_dist`
5. `edge_support`
6. `edge_support_id`
7. `dir_x`
8. `dir_y`
9. `dir_z`
10. `vec_x`
11. `vec_y`
12. `vec_z`
13. `segment`
