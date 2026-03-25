# Research Plan: Semantic Boundary Conditioned Geometric Field on PTv3 SemSeg

## 1. 项目目标

本项目的目标是在 Pointcept 现有语义分割版 PTv3 基线之上，扩展一个“语义边缘的条件几何场预测”任务。

该任务的边缘定义明确限定为语义边界，而不是任意几何突变边缘、法向突变边缘或纯曲率边缘。

输入信息默认至少包含：

- `xyz` 坐标信息
- `normal` 法向信息
- Pointcept 当前流程已允许且现有配置已在使用的其他特征，例如 `color`

需要特别说明：

- 在当前 `semseg-pt-v3m1-0-base.py` 基线中，`coord/grid_coord` 以空间字段形式进入模型流程。
- 当前 `Collect(feat_keys=("color", "normal"))` 会将 `color + normal` 拼接为 `feat`，因此当前 `in_channels=6` 对应的是特征通道，不等于把 `xyz` 直接拼进 `feat`。
- 后续开发应优先复用 Pointcept 现有的 `coord + feat` 输入机制，而不是一开始重写输入范式。

## 2. 范围约束

当前阶段的硬约束如下：

- backbone 基线固定为 Pointcept 的语义分割版 PTv3
- 配置基线固定为 Pointcept 中的 `configs/s3dis/semseg-pt-v3m1-0-base.py`
- 当前阶段不引入 KNN、EdgeConv、额外局部图模块
- 当前阶段不做大规模架构重写
- 当前阶段只做最小侵入式扩展

约束原因：

- 显存优先，先避免额外局部图结构带来的显存与算力开销
- Pointcept 工程体量较大，必须严格控制阅读和改动范围
- 当前首要目标是形成最小可运行闭环，而不是一次性追求复杂结构
- 基线越稳定，后续多头任务的误差来源越容易定位

当前明确不做：

- 不重写 PTv3 backbone
- 不改 Pointcept 的主序列化思路
- 不提前引入复杂边界先验模块
- 不在第一阶段引入复杂正则项或额外后处理系统

## 3. 任务定义

本任务定义为：

**语义边缘的条件几何场预测**

目标是在语义分割主任务之外，为每个点预测围绕语义边界构造的条件几何场。当前仓库版本已经收敛到以下三类语义：

### 3.1 support

- 定义：点执行 boundary snapping / boundary projection 的连续支撑权重
- 来源：`edge.npy[:, 3]`
- 语义：越靠近真实边缘支撑元越高，远离后衰减到 `0`

### 3.2 vec

- 定义：点到最近边缘支撑元投影点的位移向量
- 来源：`edge.npy[:, 0:3]`
- 语义：这是当前边缘分支的主几何监督

### 3.3 valid

- 定义：当前点是否存在数值稳定且位于监督半径内的最近支撑元
- 来源：`edge.npy[:, 4]`
- 语义：只作为数值有效域，不是预测目标

统一约束：

- 当前训练侧预测 `support + vec`
- `valid` 只用于有效域裁剪
- 旧 `mask / strength` 表述只保留为兼容别名，不再是主任务定义

## 4. 方法设计（当前版本）

当前版本采用轻量多头结构，保持最小侵入：

- `semantic head`
- `boundary support head`
- `boundary offset head`

设计原则：

- backbone 固定为 Pointcept PTv3 语义分割 backbone，共享使用，只负责特征提取
- backbone 继续复用 PTv3 语义分割主干
- 不改变 Pointcept 当前基于序列化与稀疏表示的主流程
- 不重写 PTv3 backbone，不改主干结构
- 尽量复用现有语义分割实现
- 边界场部分作为增量 head 扩展，而不是重写 backbone

当前推荐的结构边界：

- 保留现有语义分割模型外壳的接口风格
- 语义分支保持与当前 baseline 一致
- 边缘任务作为新增分支处理
- 后续新增工作优先集中在“语义分割模型壳层 + 新增边缘 head + 汇总 loss”
- PTv3 backbone 本身应尽量保持不动

当前阶段先确认的架构边界是：

- 共享 backbone
- 保持语义分支
- 新增边缘 head

具体新增分支的精细形式以及 loss 权重细节，不在本阶段定稿。

## 5. loss 设计草案

当前计划中的 loss 组成如下：

- semantic segmentation loss
- support regression loss
- support overlap loss
- support-weighted vec loss
- 可选 smooth regularization

约束说明：

- `semantic segmentation loss` 继续作为主任务损失
- 语义分支对应的 loss 设计当前保持不变，继续沿用现有语义分割主任务损失
- `support` 是主连续监督
- `vec loss` 按 `support * valid` 做软加权
- `valid` 只裁剪数值有效域，不单独作为预测目标
- `smooth regularization` 仅作为后续候选项，不在第一阶段强行实现

第一开发阶段的 loss 原则：

- 先实现最小闭环版本
- 最小闭环版本不强绑复杂正则
- 正则化属于增强项，而不是第一阶段阻塞项

当前边缘监督约束补充如下：

- 不是所有点都参与有效边缘几何监督
- `valid` 只负责过滤数值无效点
- `vec` 的监督权重由 `support * valid` 决定
- 后续若继续细化 loss 或 head，也应围绕 `support / vec / valid` 语义展开

## 6. 开发阶段拆分

### Phase 0

确认最小相关代码范围，编写研究计划与协作文档，建立独立项目仓库骨架。

### Phase 1

确认数据接口与监督字段接入方案。

### Phase 2

为 PTv3 语义分割基线增加多头输出。

### Phase 3

接入基础 loss，跑通训练。

### Phase 4

增加日志、可视化与消融。

### Phase 5

评估是否加入 smooth regularization 等增强项。

### 当前进度状态

截至当前仓库版本，以下内容已经具备最小可用实现：

- 独立仓库内的数据接入与 edge 同步主线
- 共享 backbone + 语义分支 + 边界分支的最小模型壳层
- 最小 loss 闭环
- 最小 validation evaluator
- 项目内 trainer、checkpoint、resume / weight、scheduler
- 单卡场景下更接近 Pointcept 的训练组织

当前尚未完成的部分主要包括：

- 项目内 test pipeline
- 结果导出与可视化
- 更完整的训练运行系统能力
- 更复杂的边界任务设计细化

## 7. 工程约束

后续开发必须遵守以下工程约束：

- 优先做最小改动
- 优先复用现有 Pointcept 训练与配置体系
- 每一步都要记录改动文件、改动目的、潜在风险
- 避免一次性大改
- 新增代码尽量集中，方便回退
- 文档必须持续维护，每阶段结束后更新

推荐执行方式：

- 每一轮只解决一个清晰问题
- 每一轮改动前先确认影响文件集合
- 若某一步需要扩大阅读范围，必须先说明扩大原因

## 8. Agent 协作约定

### Developer

职责：

- 根据 prompt 实现最小必要代码改动
- 严格遵守范围约束
- 尽量把改动集中在新增头部、字段接入和 loss 汇总层

### Reviewer

职责：

- 检查代码正确性
- 检查改动是否过度侵入
- 检查实现是否可维护
- 对下一步提出约束内建议

### Maintainer

职责：

- 在每轮结束后更新本文件
- 维护当前架构状态、开发范围、待办和风险
- 确保文档持续反映“现在做什么 / 现在不做什么”

## 9. 风险与注意事项

当前至少存在以下风险：

- Pointcept 仓库较大，容易误读无关模块，导致开发范围失控
- 多头任务可能影响语义分割主任务收敛
- `mask / direction / magnitude` 的监督定义必须与数据预处理严格一致
- 第一步若过早引入复杂结构，会显著提高调试成本
- 若字段命名、shape 约定、归一化方式不稳定，后续训练与可视化会反复返工

当前应对策略：

- 先收敛最小相关代码范围
- 先确认数据字段与监督定义
- 先保证最小训练闭环，再做增强

## 10. 当前结论与下一步建议

当前推荐的最小实现路线：

1. 固定 Pointcept 中的 `semseg-pt-v3m1-0-base.py` 作为唯一主基线。
2. 先梳理数据字段接入，不立刻改 backbone。
3. 保持 PTv3 backbone 只承担共享特征提取，不重写主干。
4. 保持语义分支与当前 baseline 一致。
5. 后续在语义分割模型外壳上增加新增边缘 head，而不是改 backbone。
6. 保持训练主线只消费统一 loss 的接口约定。
7. 在最小训练闭环稳定前，不引入复杂正则或额外局部图模块。

下一步最应该做的事情：

**优先在当前最小可训练系统的基础上继续补齐运行体系能力与 test 闭环，再决定更细粒度的边界分支形式与 loss 细化。**
