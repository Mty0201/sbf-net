# BF Data Format

## 1. 文档目的

本文档定义当前阶段 `BF (Boundary Field)` 数据集的基础语义分割数据格式。

当前目标仅为：

- 以真实 BF 数据目录结构为准
- 对齐 Pointcept 现有语义分割数据主线
- 为后续 BF 基础字段接入做好文档准备

当前不包含：

- edge 字段接入实现
- dataset 实现
- config 实现
- 多任务字段设计

## 2. 数据根目录

推荐数据根目录表示为：

`/path/to/BF_edge_chunk_npy`

后续所有 BF 基础字段接入讨论，均以该目录结构为准。实际运行路径通过环境变量 `SBF_DATA_ROOT` 指定，不依赖机器本地绝对路径。

## 3. Split 约定

当前真实 split 名称必须统一为磁盘实际名称：

- `training`
- `validation`

当前阶段不使用以下命名作为正式 split：

- `train`
- `val`
- `test`

如果后续需要测试集，应在未来阶段单独定义；本阶段文档只覆盖当前已存在且已确认的 `training / validation`。

## 4. BF 基础语义分割数据结构

推荐按如下方式理解当前 BF 基础数据集：

```text
/path/to/BF_edge_chunk_npy/
├── training/
│   ├── 020101/
│   ├── 020102/
│   └── ...
└── validation/
    ├── 010101/
    ├── 010102/
    └── ...
```

约束说明：

- `training/` 下每个子目录表示一个训练样本
- `validation/` 下每个子目录表示一个验证样本
- 每个样本目录通过其中的 `.npy` 文件表达点级字段

## 5. 单样本目录结构

当前真实单样本目录结构可抽象为：

```text
sample_id/
├── coord.npy
├── color.npy
├── normal.npy
├── segment.npy
└── edge.npy
```

示例：

```text
training/020101/
├── coord.npy
├── color.npy
├── normal.npy
├── segment.npy
└── edge.npy
```

当前阶段只把以下字段视为 BF 基础语义分割字段：

- `coord.npy`
- `color.npy`
- `normal.npy`
- `segment.npy`

`edge.npy` 当前属于“已存在事实记录”，但不属于本轮接入范围。

## 6. 基础字段定义

### 6.1 coord.npy

- 字段名：`coord`
- 推荐 shape：`(N, 3)`
- 推荐 dtype：`float32`
- 含义：点云坐标，对应每个点的 `x, y, z`

### 6.2 color.npy

- 字段名：`color`
- 推荐 shape：`(N, 3)`
- 推荐 dtype：`float32`
- 含义：点云颜色，通常按 RGB 存储

### 6.3 normal.npy

- 字段名：`normal`
- 推荐 shape：`(N, 3)`
- 推荐 dtype：`float32`
- 含义：点法向

### 6.4 segment.npy

- 字段名：`segment`
- 推荐 shape：`(N,)` 或 `(N, 1)`
- 推荐 dtype：`int32`
- 含义：语义分割标签

统一约束：

- `coord / color / normal / segment` 的第一维 `N` 必须严格一致
- 当前阶段 BF 基础语义分割只依赖这四个字段

## 7. edge.npy 的当前边界

当前真实数据中已经存在：

- `edge.npy`

但本阶段明确约束如下：

- `edge.npy` 不属于 BF 基础语义分割字段
- 本轮不定义 `edge.npy` 的读取接口
- 本轮不定义 `edge.npy` 的 shape / dtype 作为接入规范
- 本轮不讨论如何让 `edge.npy` 进入 dataset / transform / collect 主线

当前唯一需要记录的事实是：

- `edge.npy` 在真实样本目录中存在
- 它属于后续 edge 扩展阶段的候选输入或监督来源

## 8. 最小示例

```text
/path/to/BF_edge_chunk_npy/
├── training/
│   └── 020101/
│       ├── coord.npy      # (N, 3), float32
│       ├── color.npy      # (N, 3), float32
│       ├── normal.npy     # (N, 3), float32
│       ├── segment.npy    # (N,) or (N, 1), int32
│       └── edge.npy       # 当前仅记录其存在，不纳入本轮接入
└── validation/
    └── 010101/
        ├── coord.npy
        ├── color.npy
        ├── normal.npy
        ├── segment.npy
        └── edge.npy
```

## 9. 训练/验证必须字段检查清单

在当前阶段，`training` 和 `validation` 中的每个样本目录都应至少满足以下检查项：

- 存在 `coord.npy`
- 存在 `color.npy`
- 存在 `normal.npy`
- 存在 `segment.npy`
- `coord.npy` 的 shape 为 `(N, 3)`
- `color.npy` 的 shape 为 `(N, 3)`
- `normal.npy` 的 shape 为 `(N, 3)`
- `segment.npy` 的 shape 为 `(N,)` 或 `(N, 1)`
- `coord / color / normal / segment` 的点数 `N` 一致
- `segment.npy` 为整型标签

当前阶段不把以下项列为 BF 基础接入的必检约束：

- `edge.npy` 的 shape
- `edge.npy` 的 dtype
- `edge.npy` 是否进入训练主线

## 10. 后续实现落点约定

围绕 BF 基础字段接入，后续默认落点应保持如下：

- BF 数据集相关配置：`semantic-boundary-field/configs/bf/`
- dataset 扩展代码：`semantic-boundary-field/project/datasets/`
- 数据检查脚本：`semantic-boundary-field/scripts/check_data/`

当前文档只完成格式定义，不包含这些位置上的实现内容。
