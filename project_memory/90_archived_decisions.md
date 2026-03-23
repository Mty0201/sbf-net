# Archived Decisions

- 旧版 `edge.npy` 五列 `[vec_x, vec_y, vec_z, edge_support, edge_valid]`: 已废弃，因为正式训练改用显式 `direction + distance + support + valid` 六列格式。
- 把 `edge_valid` 当作预测 mask 目标: 已废弃，因为 `edge_valid` 现在只表示监督有效域。
- 用 `edge_support` 代替距离监督: 已废弃，因为 `edge_dist` 已单独建模并单独优化。
- 把 support 设计成精确边界轮廓重建目标: 已废弃，因为当前 support 只承担覆盖优先的边界邻域提案。
- 以 `loss_mask / loss_strength` 作为主命名: 已废弃，因为当前主命名固定为 `support / direction / distance`。
- 以 Pointcept trainer 作为运行入口: 已废弃，因为当前运行入口固定为项目内 trainer。
