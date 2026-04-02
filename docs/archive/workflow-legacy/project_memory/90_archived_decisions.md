# Archived Decisions

- 历史 edge loss 主线并非无效尝试: 早期设计曾带来约 `+1` mIoU 收益；其中 `reg` 思路与当前 support_reg 本质一致，`overlap` 本质上就是当前的覆盖约束；当时尚未拆成 `direction / distance`，而是直接预测连续位移向量场 `vec`。该历史结论只作为当前阶段判断的背景，不改变当前“已进入 `Stage-2 architecture rollout / verification phase`、以 `axis + side + support` 核证新主线”的核心目标。
- 继续把旧 signed-direction 表达 `support + dir + dist` 当作当前 active 主表达: 已废弃；它现在只作为历史对照和已确认失败路线存在，当前 active 主线是 `axis + side + support`。
- 旧版 `edge.npy` 五列 `[vec_x, vec_y, vec_z, edge_support, edge_valid]`: 已废弃，因为正式训练改用显式 `direction + distance + support + valid` 六列格式。
- 把 `edge_valid` 当作预测 mask 目标: 已废弃，因为 `edge_valid` 现在只表示监督有效域。
- 用 `edge_support` 代替距离监督: 已废弃，因为 `edge_dist` 已单独建模并单独优化。
- 把 support 设计成精确边界轮廓重建目标: 已废弃，因为当前 support 只承担覆盖优先的边界邻域提案。
- 以 `loss_mask / loss_strength` 作为主命名: 已废弃，因为当前 active 主命名固定为 `axis / side / support`；历史对照路线使用 `support / direction / distance`。
- 以 Pointcept trainer 作为运行入口: 已废弃，因为当前运行入口固定为项目内 trainer。
- 把 `support_cover` 作为主项、`support_reg` 作为弱辅助: 已废弃，因为当前 sweep 已确认更合理的组合是 `support_reg=1.0` 主导、`support_cover≈0.25` 弱辅助。
- 在当前阶段继续围绕 `dist` 或历史 `vec` 做细碎扫参: 已收束，因为目前没有证据表明它们是当前主矛盾。
