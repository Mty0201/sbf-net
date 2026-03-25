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

兼容别名：
- `edge_mask`
- `edge_strength`

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
- 按 `semantic_pair` 分组后做 coarse spatial clustering。
- 对 coarse cluster 做轻量去噪和 trigger 标记。

依赖输入：
- `boundary_centers.npz`

必要输出：
- `local_clusters.npz`

可视化输出：
- `clustered_boundary_centers.xyz`

这些可视化主要看什么：
- 看 surviving centers 是否形成合理的局部边界簇。
- 看 cluster id 是否局部连续。
- 看 trigger cluster 是否主要出现在方向混杂或结构更复杂的位置。


### 2.3 `scripts/fit_local_supports.py`

作用：
- 从 `local_clusters` 重建 cluster record。
- 对普通 cluster 拟合 line / polyline support。
- 对 trigger cluster 先做组内重组，再拟合 support。

依赖输入：
- `boundary_centers.npz`
- `local_clusters.npz`

必要输出：
- `supports.npz`

可视化输出：
- `support_geometry.xyz`
- `trigger_group_classes.xyz`

这些可视化主要看什么：
- `support_geometry.xyz`：看 support 几何是否沿真实边界展开，是否有明显串线、短碎片或方向错误。
- `trigger_group_classes.xyz`：看 trigger cluster 内部的主边组、碎片组和坏组划分是否基本合理。


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

兼容导出：
- `edge_mask.npy`
- `edge_strength.npy`

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
- 兼容保留 `edge_mask.npy`
- 兼容保留 `edge_strength.npy`


### 3.3 各阶段保留的可视化输出

第一阶段：
- `boundary_candidates.xyz`
- `boundary_centers.xyz`

第二阶段：
- `clustered_boundary_centers.xyz`

第三阶段：
- `support_geometry.xyz`
- `trigger_group_classes.xyz`

第四阶段：
- `edge_supervision.xyz`


## 4. 推荐运行顺序

下面是单场景推荐运行顺序。将路径替换成自己的场景目录即可。

假设：
- 原始场景目录：`/path/to/scene`
- 阶段输出目录：`/path/to/output`

### 第一步：构建 boundary centers

```bash
python pointcept/datasets/preprocessing/bf_edge_v3/scripts/build_boundary_centers.py \
  --scene /path/to/scene \
  --output /path/to/output
```

### 第二步：构建 local clusters

```bash
python pointcept/datasets/preprocessing/bf_edge_v3/scripts/build_local_clusters.py \
  --input /path/to/output \
  --output /path/to/output
```

### 第三步：拟合 local supports

```bash
python pointcept/datasets/preprocessing/bf_edge_v3/scripts/fit_local_supports.py \
  --input /path/to/output \
  --output /path/to/output
```

### 第四步：构建 pointwise edge supervision

```bash
python pointcept/datasets/preprocessing/bf_edge_v3/scripts/build_pointwise_edge_supervision.py \
  --input /path/to/output \
  --output /path/to/output \
  --max-edge-dist 0.08 \
  --ignore-index -1
```

说明：
- 第一阶段的 `--scene` 指向原始场景目录。
- 第二到第四阶段的 `--input` 指向前一阶段产物所在目录。
- 当前没有统一批处理入口；若需要批处理，建议先按这四步确认单场景结果稳定。

## 4.1 两步式数据集级批处理

第一步：原地为样本补 `supports.npz + support_geometry.xyz`

```bash
python pointcept/datasets/preprocessing/bf_edge_v3/scripts/build_support_dataset_v3.py \
  --input /path/to/dataset
```

第二步：构建新的紧凑 edge 数据集

```bash
python pointcept/datasets/preprocessing/bf_edge_v3/scripts/build_edge_dataset_v3.py \
  --input /path/to/dataset \
  --output /path/to/dataset_edge \
  --max-edge-dist 0.08 \
  --ignore-index -1
```


## 5. 可视化验收说明

### 5.1 boundary_centers 阶段看什么

- `boundary_candidates.xyz` 是否主要贴着语义边界分布。
- 是否出现大面积内部点被错误标成候选点。
- `boundary_centers.xyz` 是否沿边界连续，而不是散乱漂移。
- 切向和法向是否明显失稳。


### 5.2 local_clusters 阶段看什么

- `clustered_boundary_centers.xyz` 中同一边界带是否被分成合理局部簇。
- 是否出现明显跨边界误合并。
- trigger 标记是否主要集中在复杂区域，而不是普遍泛化。


### 5.3 supports 阶段看什么

- `support_geometry.xyz` 中 support 是否贴着真实边界延展。
- 是否存在明显过短、碎裂、串线或方向错误的 support。
- `trigger_group_classes.xyz` 中 trigger cluster 的内部分类是否大致符合直觉。


### 5.4 pointwise 阶段看什么

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

各字段怎么理解：
- `edge_dist`：点到最近 support 的距离。
- `edge_support`：由 `edge_dist` 通过监督半径内的截断高斯映射得到的吸附权重。
- `edge_support_id`：当前点命中的最近 support id。
- `edge_dir`：从点指向最近边界的单位方向。
- `edge_vec`：从点指向最近边界的位移向量。
- `segment`：原始语义标签，方便按类别检查监督带。

额外说明：
- `edge_valid` 不单独写入 `edge_supervision.xyz`，因为这里本身只导出 `edge_valid == 1` 的点。

重点关注的异常现象：
- 有效点没有沿真实边界形成连续带状区域。
- `edge_support_id` 在局部区域频繁跳变。
- `edge_dir` 或 `edge_vec` 出现明显随机方向。
- `edge_support` 不是“边界附近强、远离边界弱”，而是分布突兀或断裂。
- 某些类别附近出现明显错误吸附到远处 support。


## 6. 当前实现边界与注意事项

当前版本已经做到：
- 完成四阶段逐步构建。
- 能从语义点云生成 pointwise edge supervision。
- 保留了每个阶段的人工验收可视化输出。

当前版本还没有做：
- 语义法向替换。
- smoothing / diffusion。
- 统一的全量批处理入口。
- 研究性对比分析。

当前 pointwise supervision 的关键边界：
- 当前逐点监督基于最近 support 查询，而不是直接对原始边界几何做解析。
- `edge_support` 使用监督半径内的截断高斯衰减。
- `sigma` 由 `support_radius / 2` 推导。
- 超过 `support_radius` 或无合法候选 support 的点，`edge_valid = 0`。
- 当前 `edge_supervision.xyz` 只导出有效点，用于减小体积并方便 CloudCompare 验收。


## 7. 使用建议

- 先在单场景上按四阶段顺序跑通，再考虑批处理。
- 每个阶段都建议查看对应 `.xyz` 可视化，不要只看最终数组。
- 若 pointwise 结果异常，优先回溯检查 `supports`，再检查 `local_clusters`，最后检查 `boundary_centers`。
- 当前流程的最终目标是逐点监督质量，不是让中间支撑元在几何上“看起来更漂亮”。
