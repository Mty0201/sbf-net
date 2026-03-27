# Data Pipeline

- 当前训练样本目录: `coord.npy`, `color.npy`, `normal.npy`, `segment.npy`, `edge.npy`。
- scene 读取入口: `load_scene()` 读取 `coord.npy`、`segment.npy`，可选读取 `color.npy`、`normal.npy`。
- support 构建主链: `build_support_dataset_v3.py` 内部顺序为 `boundary_centers -> local_clusters -> supports.npz`。
- pointwise 监督生成: `build_pointwise_edge_supervision.py` 基于 `scene + supports.npz` 生成逐点 `edge_vec / edge_dist / edge_dir / edge_support / edge_valid`。
- 紧凑数据集导出: `build_edge_dataset_v3.py` 复制基础字段并导出旧版 `edge.npy = [vec_x, vec_y, vec_z, edge_support, edge_valid]`。
- 正式训练 GT 转换: `convert_edge_vec_to_dir_dist.py` 把旧版 `edge.npy` 转为当前正式格式 `[dir_x, dir_y, dir_z, edge_dist, edge_support, edge_valid]`。
- 当前训练读取: `BFDataset.get_data()` 直接加载样本目录中的正式 `edge.npy`；缺失 `edge.npy` 视为数据错误并显式失败。
- train transform: `InjectIndexValidKeys(edge)` -> Pointcept augmentations -> `GridSample` -> `SphereCrop` -> `ToTensor` -> `Collect`。
- val transform: `InjectIndexValidKeys(edge)` -> `GridSample(return_inverse=True)` -> `ToTensor` -> `Collect`。
- GT/support: `edge_support` 由 `edge_dist` 在 `support_radius` 内生成截断 Gaussian。
- GT/support 参数: `sigma = support_radius / 2`。
- GT/direction: `edge_dir = normalize(edge_vec)`。
- GT/direction 零值规则: `dist < eps` 时方向写为零向量。
- GT/distance: `edge_dist = ||edge_vec||`。
- GT/valid: `edge_valid = 1` 当点找到最近 support 且距离 `<= support_radius`，否则为 `0`。
