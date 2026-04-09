# BF Edge v3 Pipeline

## 1. 流程总目标

`bf_edge_v3` 的目标是从建筑立面语义点云出发，逐阶段构建逐点边缘监督，而不是把中间支撑元本身当作最终目标。

当前流程分为四个阶段：

1. `boundary centers`
2. `local clusters`
3. `local supports`
4. `pointwise edge supervision`

最终输出的逐点监督信号包括：

- `edge_dist`
- `edge_dir`
- `edge_valid`
- `edge_support_id`
- `edge_vec`
- `edge_support`

中间的 `boundary_centers / local_clusters / supports` 都只是为最终 pointwise supervision 服务的中间表示。


## 2. 四阶段脚本说明

### 2.1 `scripts/build_boundary_centers.py`

作用：
- 从原始语义点云中检测语义边界候选点。
- 为候选点构造 `boundary centers`，作为后续聚类输入。

依赖输入：
- `coord.npy`
- `segment.npy`
- `normal.npy`：可选，切向估计的 fallback 会用到
- `color.npy`：可选，仅用于兼容场景读取，不影响主流程

必要输出：
- `boundary_centers.npz`

可视化输出：
- `boundary_candidates.xyz`
- `boundary_centers.xyz`

这些可视化主要看什么：
- `boundary_candidates.xyz`：看候选点是否主要落在语义边界附近，是否出现大面积非边界误检。
- `boundary_centers.xyz`：看中心点位置、法向和切向是否稳定，是否沿真实边界连续分布。


### 2.2 `scripts/build_local_clusters.py`

作用：
- 读取 `boundary_centers`。
- 按 `semantic_pair` 分组后做 bottom-up micro-cluster merge：
  1. 小 eps DBSCAN 生成紧密微簇
  2. 双峰横向分裂：分离被连接边桥接的平行边
  3. 方向感知合并：相邻且切向兼容的微簇通过 union-find 合并
  4. 后合并救援：噪声点分配到最近已合并簇

依赖输入：
- `boundary_centers.npz`

必要输出：
- `local_clusters.npz`

可视化输出：
- `clustered_boundary_centers.xyz`

这些可视化主要看什么：
- 看 surviving centers 是否形成合理的局部边界簇。
- 看 cluster id 是否局部连续。
- 看是否出现跨边界误合并。


### 2.3 `scripts/fit_local_supports.py`

作用：
- 从 `local_clusters` 重建 cluster record。
- 对每个 cluster 做空间间隙分割后拟合 line / polyline support。
- 所有 cluster 统一处理，无 trigger dispatch。

依赖输入：
- `boundary_centers.npz`
- `local_clusters.npz`

必要输出：
- `supports.npz`

可视化输出：
- `support_geometry.xyz`

这些可视化主要看什么：
- `support_geometry.xyz`：看 support 几何是否沿真实边界展开，是否有明显串线、短碎片或方向错误。


### 2.4 `scripts/build_pointwise_edge_supervision.py`

作用：
- 读取场景点和已有 `supports`。
- 对每个点只在 `semantic_pair` 包含该点标签的 support 中找最近 support。
- 生成逐点边缘监督字段。

依赖输入：
- `coord.npy`
- `segment.npy`
- `supports.npz`

必要输出：
- `edge_dist.npy`
- `edge_dir.npy`
- `edge_valid.npy`
- `edge_support_id.npy`
- `edge_vec.npy`
- `edge_support.npy`

可视化输出：
- `edge_supervision.xyz`

这些可视化主要看什么：
- 看有效监督点是否沿真实边界形成稳定边界带。
- 看 `edge_support_id` 是否局部连续。
- 看 `edge_dir / edge_vec` 是否整体指向最近边界。
- 看 `edge_support` 是否在边界附近更强、向两侧平滑衰减。


## 3. 输入输出约定

### 3.1 初始输入

流程起点至少要求：

- `coord.npy`
- `segment.npy`

可选输入：

- `normal.npy`
- `color.npy`

前置条件：
- 点云已经有逐点语义标签。
- `segment.npy` 中的标签应与后续边界检测语义一致。


### 3.2 各阶段核心输出

第一阶段：
- `boundary_centers.npz`

第二阶段：
- `local_clusters.npz`

第三阶段：
- `supports.npz`

第四阶段：
- `edge_dist.npy`
- `edge_dir.npy`
- `edge_valid.npy`
- `edge_support_id.npy`
- `edge_vec.npy`
- `edge_support.npy`


### 3.3 各阶段保留的可视化输出

第一阶段：
- `boundary_candidates.xyz`
- `boundary_centers.xyz`

第二阶段：
- `clustered_boundary_centers.xyz`

第三阶段：
- `support_geometry.xyz`

第四阶段：
- `edge_supervision.xyz`


## 4. 推荐运行方式

### 4.1 一步式批处理（推荐）

`rebuild_edge_dataset_inplace.py` 是主力批处理入口，在内存中执行完整四阶段流程，每个场景只输出三个文件：`edge.npy`、`edge_supervision.xyz`、`support_geometry.xyz`。

**安装 C++ 加速（首次使用前执行一次）：**

```bash
cd data_pre/bf_edge_v3/cpp
pip install --no-build-isolation .
```

**批处理整个数据集：**

```bash
cd data_pre/bf_edge_v3
python scripts/rebuild_edge_dataset_inplace.py \
    --input /path/to/dataset --workers 4
```

**强制重跑已有结果的场景：**

```bash
python scripts/rebuild_edge_dataset_inplace.py \
    --input /path/to/dataset --workers 4 --force
```

启动时打印 `Backend: C++` 或 `Backend: Python` 表示当前后端。
C++ 后端约 8s/场景（367K 点），Python 后端约 200s/场景。

跳过逻辑：如果场景已有 `edge.npy`（shape `(N, 5)`）+ `edge_supervision.xyz` + `support_geometry.xyz`，且未指定 `--force`，则跳过。


### 4.2 单场景逐阶段运行（调试用）

将路径替换成自己的场景目录即可。

```bash
cd data_pre/bf_edge_v3

# 第一步：构建 boundary centers
python scripts/build_boundary_centers.py \
    --scene /path/to/scene --output /path/to/output

# 第二步：构建 local clusters
python scripts/build_local_clusters.py \
    --input /path/to/output --output /path/to/output

# 第三步：拟合 local supports
python scripts/fit_local_supports.py \
    --input /path/to/output --output /path/to/output

# 第四步：构建 pointwise edge supervision
python scripts/build_pointwise_edge_supervision.py \
    --input /path/to/output --output /path/to/output \
    --support-radius 0.08 --ignore-index -1
```


### 4.3 两步式数据集级批处理（旧流程）

第一步：原地为样本补 `supports.npz + support_geometry.xyz`

```bash
python scripts/build_support_dataset_v3.py --input /path/to/dataset
```

第二步：构建新的紧凑 edge 数据集

```bash
python scripts/build_edge_dataset_v3.py \
    --input /path/to/dataset \
    --output /path/to/dataset_edge \
    --support-radius 0.08 --ignore-index -1
```

注意：4.1 的一步式流程是推荐做法，4.3 仅在需要保留中间 `supports.npz` 时使用。


## 5. C++ 加速后端

`cpp/` 目录包含全部四阶段的 C++ 实现，通过 pybind11 暴露为 `bf_edge_cpp` Python 模块。

### 5.1 依赖

- pybind11 (pip)
- Eigen3 (conda install eigen 或系统 libeigen3-dev)
- cmake
- OpenMP (可选，用于多线程加速)

### 5.2 编译安装

```bash
cd data_pre/bf_edge_v3/cpp
pip install --no-build-isolation .
```

验证：

```bash
python -c "import bf_edge_cpp; print(dir(bf_edge_cpp))"
```

### 5.3 性能对比（367K 点场景）

| Stage | Python | C++ | 加速比 |
|-------|--------|-----|--------|
| 1 (boundary centers) | 60s | 1.8s | 33x |
| 2 (clustering) | 5s | 0.45s | 11x |
| 3 (support fitting) | 2.4s | 0.35s | 7x |
| 4 (pointwise supervision) | 133s | 5.8s | 23x |
| **合计** | **200s** | **8.4s** | **24x** |

### 5.4 自动降级

`rebuild_edge_dataset_inplace.py` 在启动时尝试 `import bf_edge_cpp`。
如果导入失败，自动回退到纯 Python 实现，功能完全相同。


## 6. Module Structure

### 6.1 Core Module Inventory

| Module | Lines | Stage | Responsibility |
|--------|-------|-------|----------------|
| `core/boundary_centers_core.py` | 414 | 1 | kNN boundary detection, center estimation, confidence scoring |
| `core/local_clusters_core.py` | 723 | 2 | Bottom-up micro-cluster merge: DBSCAN, bimodal split, direction merge, rescue |
| `core/supports_core.py` | 338 | 3 | Orchestration: cluster record rebuild, support record assembly |
| `core/fitting.py` | 263 | 3 | Core geometry primitives, line/polyline fitting, spatial gap splitting |
| `core/supports_export.py` | 83 | 3 | NPZ/XYZ export and visualization for supports |
| `core/config.py` | 143 | 1-4 | Frozen dataclass configs: Stage1Config, Stage2Config, Stage3Config, Stage4Config |
| `core/pointwise_core.py` | 428 | 4 | Per-point edge supervision computation, bad support detection |
| `core/validation.py` | 448 | 1-4 | Cross-stage validation hooks |

### 6.2 Parameter Centralization

所有流程参数定义在 `core/config.py` 的 frozen dataclass 中：

- **`Stage1Config`**: boundary center detection (k, min_cross_ratio, min_side_points, ignore_index)
- **`Stage2Config`**: bottom-up micro-cluster merge (micro DBSCAN eps/min_samples, bimodal split threshold, merge radius/direction/lateral, rescue radius, min_cluster_points)
- **`Stage3Config`**: support fitting (line_residual_th, min_cluster_size, max_polyline_vertices, polyline_residual_th, min_cluster_density). `to_runtime_dict()` 生成 `build_supports_payload()` 所需的 flat dict
- **`Stage4Config`**: pointwise edge supervision (support_radius=0.08, ignore_index=-1). Computed property: `sigma`

### 6.3 C++ Module

| File | Responsibility |
|------|----------------|
| `cpp/src/stage1.cpp` | Stage 1: nanoflann kNN + OpenMP parallel boundary detection |
| `cpp/src/stage2.cpp` | Stage 2: nanoflann DBSCAN + union-find merge |
| `cpp/src/stage3.cpp` | Stage 3: Eigen SVD line/polyline fitting |
| `cpp/src/stage4.cpp` | Stage 4: OpenMP per-label parallel pointwise supervision |
| `cpp/src/bindings.cpp` | pybind11 bindings exposing 5 Python functions |
| `cpp/include/bf_edge/` | Headers: common.h, kdtree.h, stage1-4.h |
| `cpp/third_party/nanoflann/` | Vendored nanoflann header-only KD-tree |


## 7. 可视化验收说明

### 7.1 boundary_centers 阶段看什么

- `boundary_candidates.xyz` 是否主要贴着语义边界分布。
- 是否出现大面积内部点被错误标成候选点。
- `boundary_centers.xyz` 是否沿边界连续，而不是散乱漂移。
- 切向和法向是否明显失稳。


### 7.2 local_clusters 阶段看什么

- `clustered_boundary_centers.xyz` 中同一边界带是否被分成合理局部簇。
- 是否出现明显跨边界误合并。


### 7.3 supports 阶段看什么

- `support_geometry.xyz` 中 support 是否贴着真实边界延展。
- 是否存在明显过短、碎裂、串线或方向错误的 support。


### 7.4 pointwise 阶段看什么

`edge_supervision.xyz` 当前只包含 `edge_valid == 1` 的点，不包含无效点。

字段顺序为：

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

重点关注的异常现象：
- 有效点没有沿真实边界形成连续带状区域。
- `edge_support_id` 在局部区域频繁跳变。
- `edge_dir` 或 `edge_vec` 出现明显随机方向。
- `edge_support` 不是"边界附近强、远离边界弱"，而是分布突兀或断裂。
- 某些类别附近出现明显错误吸附到远处 support。


## 8. 使用建议

- **批处理推荐** `rebuild_edge_dataset_inplace.py --workers 4`，一步完成全部四阶段。
- **调试排查** 时按四阶段顺序逐步跑，每步查看对应 `.xyz` 可视化。
- 若 pointwise 结果异常，优先回溯检查 `supports`，再检查 `local_clusters`，最后检查 `boundary_centers`。
- 当前流程的最终目标是逐点监督质量，不是让中间支撑元在几何上"看起来更漂亮"。


## 9. 当前实现边界与注意事项

当前 pointwise supervision 的关键边界：
- 当前逐点监督基于最近 support 查询，而不是直接对原始边界几何做解析。
- `edge_support` 使用监督半径内的截断高斯衰减。
- `sigma` 由 `support_radius / 2` 推导。
- 超过 `support_radius` 或无合法候选 support 的点，`edge_valid = 0`。
- 当前 `edge_supervision.xyz` 只导出有效点，用于减小体积并方便 CloudCompare 验收。
