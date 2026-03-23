# Current Architecture

- 主模型: `SharedBackboneSemanticBoundaryModel`。
- 主干结构: `PT-v3m1` shared backbone -> `SemanticHead` + `EdgeHead`。
- semantic 分支: 输出 `seg_logits`，负责 8 类语义分割。
- edge 分支: 输出 `dir_pred(3) + dist_pred(1) + support_pred(1)`。
- trainer/loss 统一入口: `edge_pred = [dir, dist, support]`。
- semantic: 主任务输出，负责类别预测。
- support: 粗边界邻域场，表示点处于有效边界吸附带内的强度。
- direction: 单位吸附方向，表示点指向最近 support 的方向。
- distance: 吸附步长，表示点到最近 support 的物理距离。
- valid: 只存在于 GT，表示该点是否进入有效监督域。
- 关系: semantic 负责类别，support 定义边界邻域，direction + distance 共同定义几何吸附场，valid 只控制监督范围。
- 数据模块: `BFDataset` 负责加载样本目录中的 `edge.npy`。
- 变换模块: `InjectIndexValidKeys` 负责把 `edge` 纳入 Pointcept 的索引同步链。
- loss 模块: `SemanticBoundaryLoss` 负责 semantic + edge 联合优化。
- evaluator 模块: `SemanticBoundaryEvaluator` 负责 semantic 指标和 edge 指标统计。
- runtime 模块: `SemanticBoundaryTrainer` 负责训练、验证、scheduler、checkpoint。
