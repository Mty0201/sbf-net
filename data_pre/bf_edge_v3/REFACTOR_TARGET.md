# REFACTOR_TARGET

## 1. 项目当前状态总结

`bf_edge_v2` 当前是一个面向建筑立面点云的语义边界预处理流水线。它从带语义标签的点云出发，先检测语义边界候选点，再构造边界中心样本，对这些中心样本进行局部聚类与细分，之后拟合局部支撑元，最后基于支撑元为原始点云生成逐点的距离场和方向场监督。

当前代码中的真实 pipeline 不是简单的“边界检测 -> 聚类 -> 拟合”，而是：

1. 场景读取
2. boundary candidates 检测
3. boundary centers 构造
4. 按 semantic pair 分组
5. 空间 DBSCAN 粗聚类
6. stable kernel analysis
7. 局部连通切分
8. transition attach
9. cluster merge
10. post RANSAC split
11. support 拟合
12. pointwise vector field 生成

当前最终产物已经包含 `edge_dist.npy`、`edge_dir.npy`、`edge_mask.npy` 和 `edge_support_id.npy`，说明系统已经具备“逐点边界监督生成”能力。但从整体设计上看，系统依然强依赖 `boundary_centers`、`local_clusters`、`supports` 这些中间几何表达；最终逐点场并不是主设计中心，而更像是在已有支撑元表示之上追加的一层派生结果。

换句话说，当前设计路径更接近：

“先把边界组织成局部几何支撑元，再由支撑元反推逐点监督”

而不是：

“围绕最终逐点距离场和方向场，选择必要且最稳定的中间表示”


## 2. 重构后的目标定义

重构后的端到端目标需要重新收敛到逐点边界场本身。

输入保持不变。场景目录仍然至少包含：

- `coord.npy`
- `segment.npy`
- `normal.npy`
- `color.npy`

端到端核心输出改为逐点边界监督场，即每个点相对于最近语义边界的：

- `distance`
- `direction`
- 必要时的 `mask`
- 必要时的辅助索引或调试信息

中间表示仍然可以存在，但它们的地位需要重新定义：

- `boundary candidates`
- `boundary centers`
- `local clusters`
- `supports`

这些都只是为了帮助生成更稳定、更合理的逐点场输出，而不再是目标本身。是否保留某种中间表示，应该取决于它是否显著提升最终逐点边界距离场和方向场的质量，而不是因为它在几何上“看起来完整”或者“便于拟合支撑元”。

重构后的系统应优先围绕以下目标设计：

- 边界距离场是否贴近真实语义边界
- 边界方向场是否稳定、连续、局部一致
- 无效区域与有效区域的划分是否合理
- 输出是否适合作为后续监督信号使用

而不是优先围绕：

- support 是否拟合得漂亮
- cluster 是否像人工几何草图
- 中间几何表达是否完备


## 3. 当前结构中的关键问题

### 3.1 对 supports / local_clusters 依赖过重

当前 `build_vector_field.py` 并不是直接从边界局部信息生成逐点场，而是强依赖 `supports.npz`。这意味着最终逐点场的质量，高度受制于前面聚类和拟合阶段的设计。只要 cluster 划分不合理，或者 support 拟合不稳定，最终的 `edge_dist` 和 `edge_dir` 就会被连带影响。

这种依赖链会带来两个后果：

1. 中间几何表示会反过来绑架最终输出设计  
   本来应该围绕“如何生成更好的逐点场”来设计 pipeline，但现在很多逻辑首先是在解决“如何得到可拟合的 cluster / support”。

2. 调试路径被拉长  
   最终场质量变差时，问题可能来自候选边界、中心样本、DBSCAN、细分、merge、RANSAC 拆分、support 类型选择、投影打分等多个环节，定位成本高。

### 3.2 clustering + splitting 是当前最复杂也最值得重构的部分

`boundary centers` 之后的部分是当前复杂度最高的区域。实际逻辑已经不是单纯聚类，而是一整套围绕“得到更适合拟合的局部支撑元”构建的复杂控制流：

- semantic pair grouping
- DBSCAN
- stable kernel analysis
- 连通切分
- transition attach
- merge
- post RANSAC split

这部分复杂度高，说明这里既承担了“组织边界点”的任务，也承担了“清理不稳定点”和“让几何拟合更容易”的任务。职责已经明显混杂。

### 3.3 当前流程更像“为几何拟合服务”

从当前代码组织看，很多判断标准本质上都是在回答：

- 这个 cluster 是否足够线性
- 切向是否一致
- 是否适合拟合 line
- 不适合 line 时是否退到 polyline

这些标准对生成 support 很有帮助，但不一定与最终逐点距离场、方向场的质量完全一致。某些局部 cluster 即使不适合拟合成漂亮的 line/polyline，也可能依然足以提供稳定的局部边界方向和距离监督。反过来，一个几何上“能拟合”的支撑元，也不必然意味着它最适合生成连续、合理的逐点场。

### 3.4 职责混杂与模块耦合明显

当前职责混杂最明显的脚本是：

- `cluster_local_centers.py`
- `fit_local_supports.py`
- `build_vector_field.py`

其中：

- `cluster_local_centers.py` 同时承担粗聚类、稳定性评估、过渡点处理、cluster merge、RANSAC 再拆分和 cluster 统计输出
- `fit_local_supports.py` 同时承担 cluster 重建、支撑元选择、几何拟合、方向正则和格式导出
- `build_vector_field.py` 强依赖 support schema，并把逐点监督逻辑绑定在 support 投影方案上

耦合最强的链路是：

`boundary_centers -> local_clusters -> supports -> vector_field`

这条链路中的每一层都把下一层的设计空间压缩了，导致最终向量场模块很难独立演进。


## 4. 重构的总体原则

重构应遵循以下原则。

### 4.1 输入格式保持兼容

场景输入格式保持不变，不引入新的数据依赖，不打破已有数据组织方式。

### 4.2 输出目标明确收敛到逐点场

重构中的所有模块设计，都应以“提升逐点距离场和方向场质量”为首要目标。中间表示必须服从这一目标。

### 4.3 小步重构，不一次性推翻

当前 pipeline 已经能跑通并输出结果，因此不应直接整体推翻。应在保持可运行、可验证的前提下，逐步拆清职责、重建边界，再决定算法替换。

### 4.4 先拆职责，再改算法

当前首先要解决的是结构问题，而不是立即更换 DBSCAN 或调参数。只有当模块职责清晰后，算法层面的比较和替换才有意义。

### 4.5 先保证端到端清晰，再提升聚类/切分质量

应先明确：

- 最终要输出什么
- 每一阶段为什么存在
- 哪一步真正影响最终 field 质量

然后再针对 clustering / refinement 做定向优化。

### 4.6 每一步都可验收

每一阶段调整后，都应能回答两个问题：

1. 结构是否更清晰了
2. 最终逐点边界场是否因此更稳定、更合理


## 5. 建议的新架构方向

重构后的代码不必立刻重写为全新算法，但逻辑分层应更明确。建议从逻辑上拆成以下模块。

### 5.1 scene loading

职责：

- 统一读取 `coord/segment/normal/color`
- 校验 shape、dtype、点数一致性
- 提供统一 scene payload

这是稳定保留模块，风险低，适合尽早收敛成公共基础层。

### 5.2 boundary candidate extraction

职责：

- 基于语义邻域变化识别候选边界点
- 输出点级候选与候选置信度

这是稳定保留模块。它直接服务于边界发现，本身不依赖后续拟合策略。

### 5.3 boundary center construction

职责：

- 从候选边界点构造局部边界中心样本
- 估计局部边界法向、切向及置信度

这也是应保留的稳定模块，因为它把原始点云压缩成更适合后续组织的边界样本表示。

### 5.4 local grouping / clustering

职责：

- 将 boundary centers 组织成局部边界片段
- 明确只处理“分组/归属”问题

这是当前重构重点之一。需要从“为支撑元拟合服务的复杂混合逻辑”中抽离出真正的 grouping 问题。

### 5.5 cluster refinement / splitting

职责：

- 对局部分组结果做进一步细化
- 处理桥接、过渡、断裂、混合 cluster
- 明确细分目标是提升 field 质量，而不是单纯让几何拟合更容易

这是当前最值得重构的核心模块。它应成为独立的、可单独理解、单独调试、单独评估的层。

### 5.6 boundary representation generation

职责：

- 从 refined clusters 生成用于 pointwise field 查询的边界表示

这里的“boundary representation”不应预设必须是 `support`。可以暂时保留 support 作为一种实现，但概念上应退到“可选中间表达”。后续可比较：

- support-based representation
- cluster-direct representation
- local piecewise representation

### 5.7 pointwise field generation

职责：

- 为每个点找到合适的边界参考
- 生成 distance / direction / mask / 辅助索引

这是未来的主中心模块。它不应只是 support 的下游附属层，而应成为端到端设计的真正目标层。

### 5.8 evaluation / visualization

职责：

- 检查中间结果和最终结果
- 提供面向 field 质量的调试视图

这部分应保留，但 debug 中间结果不应绑死主流程。`boundary_centers`、`local_clusters`、`supports` 等中间文件可以保留为调试导出，而不应默认决定主系统结构。


## 6. 第一阶段重构计划

当前阶段不应立刻进入算法替换，而应先完成结构对齐。建议按以下阶段推进。

### Phase 0：明确目标与输出定义

目标：

- 明确系统的核心输出是逐点距离场和方向场
- 明确 `supports / clusters / centers` 只是中间表示
- 明确后续评价标准从“拟合质量”转向“field 质量”

交付物：

- 重构目标文档
- 输出字段定义
- 端到端目标说明

### Phase 1：整理当前 pipeline 与模块边界

目标：

- 把现有代码中的真实阶段拆清
- 明确每个阶段的输入、输出、责任和依赖
- 区分“核心流程必需模块”和“调试/可视化模块”

重点：

- 从单入口角度重新定义 pipeline
- 识别哪些中间产物是必要数据，哪些只是历史遗留或 debug 便利

### Phase 2：拆分 clustering / splitting 职责

目标：

- 把 `cluster_local_centers.py` 中混在一起的粗聚类、稳定性分析、attach、merge、post-RANSAC split 拆成逻辑独立的模块
- 明确每一步在解决什么问题，以及它是否真的服务最终 field 质量

重点：

- 让 clustering 成为“局部分组模块”
- 让 refinement/splitting 成为“局部结构修正模块”
- 降低当前一体化控制流带来的理解成本和调试成本

### Phase 3：建立面向 vector field 质量的评估标准

目标：

- 不再只看 line residual、cluster linearity 等中间指标
- 引入直接面向最终输出的质量判断

重点：

- cluster 是否带来更稳定的最近边界投影
- field 是否连续
- field 是否在局部区域内无明显跳变
- mask 是否合理覆盖真实边界邻域

### Phase 4：再考虑聚类和细分算法改进

目标：

- 在结构边界清楚后，再决定是否保留现有 DBSCAN + refinement 体系
- 再评估哪些细分逻辑需要保留、替换或删除

重点：

- 算法调整必须围绕最终 field 质量，而不是围绕中间几何形态好不好看


## 7. 后续算法改进的重点方向

后续算法改进应围绕“如何得到更稳定、合理、连续、贴近真实语义边界的逐点场”，而不是围绕“如何把 cluster 拟合得更像几何草图”。

### 7.1 聚类阶段应围绕哪些信号组织

聚类不应只围绕空间距离。更合理的组织信号包括：

- 空间位置
- `semantic_pair`
- 局部切向一致性
- 局部边界法向或分离方向
- 局部线性 / 局部可解释性
- 局部边界置信度

这些信号的目标不是把点分成“好拟合的几何段”，而是把点分成“能生成稳定边界表示的局部边界片段”。

### 7.2 细分阶段的目标究竟是什么

细分阶段不应默认理解为“把 cluster 切成更容易拟合 line/polyline 的子段”。更合理的目标应该是：

- 去除明显混合的局部边界片段
- 防止不同边界走向或不同边界面被错误混合
- 保留局部场方向的一致性
- 提高后续最近边界投影的稳定性

也就是说，细分的成功标准不应只是“簇更细了”，而应是“这个局部表示用于生成 pointwise field 时更稳定了”。

### 7.3 什么样的 cluster 才算“对最终逐点方向场有用”

一个有用的 cluster 不一定要完美线性，也不一定必须能被 line 支撑得很好。更重要的是它是否满足：

- 在局部空间上是连贯的
- 在边界语义对上是纯净的
- 在方向上没有剧烈冲突
- 能提供稳定、可解释的最近边界投影
- 不会在邻域内造成方向场跳变或距离场异常

如果一个 cluster 几何上不够漂亮，但对 pointwise field 查询稳定，它依然是有价值的。

### 7.4 从“拟合友好”转向“监督友好”

后续算法导向应逐步从：

- line/polyline 拟合友好

转向：

- 局部边界表示稳定
- 投影合理
- 方向连续
- 距离单调、可解释
- 对监督训练更友好

这意味着后续不排除继续保留 support，但 support 的评价标准要改变。它不再只是“拟合残差低”，而应看它是否真正改善了最终点级 field 的监督质量。


## 8. 当前最推荐优先做的事情

在不立刻大改算法的前提下，下一步最该优先做的事情，不是继续调 DBSCAN 参数，也不是继续修 support 拟合细节，而是：

**先做一次结构性拆分：把 `cluster_local_centers.py` 中的 clustering、refinement、splitting、merge、post-RANSAC split 明确拆成独立的逻辑层，并把它们与最终 pointwise field 目标重新对齐。**

原因很明确：

- 当前最大的复杂度和不透明性集中在这里
- 当前最可能影响最终 vector field 质量的瓶颈也集中在这里
- 如果这一层的职责不拆清，后续所有算法优化都会继续被“为 support 拟合服务”的历史结构牵着走

因此，最优先的结构性动作是：

**先重建 boundary centers 之后这段 pipeline 的模块边界和目标定义，让 clustering / refinement 成为面向最终逐点边界场质量的独立核心模块。**

在这一步完成之前，调参数的收益会非常有限，直接替换算法也会缺乏明确评价标准。
