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
- 配置基线固定为 `configs/s3dis/semseg-pt-v3m1-0-base.py`
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

目标是在语义分割主任务之外，为每个点预测围绕语义边界构造的条件几何场。当前计划包含三类监督：

### 3.1 mask

- 定义：点是否位于语义边界影响域内
- 作用：描述该点是否应该参与边界场预测
- 工程语义：既是监督标签，也可以视为一个可监督 gate

### 3.2 direction

- 定义：若点位于影响域内，预测其指向最近语义边界的方向
- 语义：这是一个有条件定义的向量量
- 约束：`direction` 只在 `mask` 内有意义

### 3.3 magnitude / intensity

- 定义：到语义边界的距离或边界强度
- 期望形式：全域定义的连续量
- 约束：远离边界时衰减，在 `mask` 外缘降为 `0`
- 明确：`magnitude` 在全域上都定义，但在 `mask` 外为 `0`

统一约束：

- `direction` 只在 `mask` 内计算监督与损失
- `magnitude` 为全域连续监督量，`mask` 外强制为 `0`
- `mask` 既用于训练监督，也为后续推理阶段提供可监督门控依据

## 4. 方法设计（当前版本）

当前版本采用轻量多头结构，保持最小侵入：

- `semantic head`
- `mask head`
- `direction head`
- `magnitude head`

设计原则：

- backbone 继续复用 PTv3 语义分割主干
- 不改变 Pointcept 当前基于序列化与稀疏表示的主流程
- 尽量复用现有语义分割实现
- 边界场部分作为增量 head 扩展，而不是重写 backbone

当前推荐的结构边界：

- 保留现有 `DefaultSegmentorV2` 的基本接口风格
- 后续新增工作优先集中在“语义分割模型壳层 + 多头输出 + 汇总 loss”
- PTv3 backbone 本身应尽量保持不动

这样做的目的不是限制最终效果，而是确保第一轮开发能把风险压缩在可控范围内。

## 5. loss 设计草案

当前计划中的 loss 组成如下：

- semantic segmentation loss
- mask loss
- direction loss
- magnitude / intensity loss
- 可选 smooth regularization

约束说明：

- `semantic segmentation loss` 继续作为主任务损失
- `mask loss` 负责学习语义边界影响域
- `direction loss` 只在 `mask` 内计算
- `magnitude / intensity loss` 作用于全域连续量，但 `mask` 外监督目标为 `0`
- `smooth regularization` 仅作为后续候选项，不在第一阶段强行实现

第一开发阶段的 loss 原则：

- 先实现最小闭环版本
- 最小闭环版本不强绑复杂正则
- 正则化属于增强项，而不是第一阶段阻塞项

## 6. 开发阶段拆分

### Phase 0

确认 Pointcept 最小相关代码范围，编写研究计划 / 开发指导文档。

当前阶段产出：

- 最小必要文件清单
- 明确当前不需要阅读的部分
- 统一的开发上游约束文档

### Phase 1

确认数据接口与监督字段接入方案。

核心目标：

- 明确监督字段落在数据集目录中的文件组织方式
- 明确哪些字段进入 transform 流程
- 明确哪些字段进入 `Collect`
- 明确训练态、验证态、测试态的字段最小差异

### Phase 2

为 PTv3 语义分割基线增加多头输出。

核心目标：

- 保留当前 backbone 主体
- 扩展最小必要 head
- 保持训练接口兼容 Pointcept 现有 trainer

### Phase 3

接入基础 loss，跑通训练。

核心目标：

- 汇总多头 loss 为统一 `output_dict["loss"]`
- 保持现有训练循环不需要结构性改写
- 先跑通最小训练闭环

### Phase 4

增加日志、可视化与消融。

核心目标：

- 分头 loss 监控
- mask / direction / magnitude 结果可视化
- 与仅语义分割基线进行对比

### Phase 5

评估是否加入 smooth regularization 等增强项。

核心目标：

- 只在最小闭环稳定后再引入增强项
- 逐项验证收益，而不是一次性叠加

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

1. 固定 `semseg-pt-v3m1-0-base.py` 作为唯一主基线。
2. 先梳理数据字段接入，不立刻改 backbone。
3. 后续在语义分割模型外壳上增加轻量多头输出。
4. 保持训练器只消费统一 `loss` 的接口约定。
5. 在最小训练闭环稳定前，不引入复杂正则或额外局部图模块。

下一步最应该做的事情：

**优先确认数据字段接入与监督组织方案，而不是立刻大改模型。**

原因：

- 多头输出能否稳定落地，首先取决于监督字段如何进入 Pointcept 的 dataset / transform / collect 主线
- 如果数据字段定义不稳，模型和 loss 设计都会反复返工

## Appendix A. 当前已确认的最小代码主线

当前仅确认了 `semseg-pt-v3m1-0-base.py` 跑通所需的最小代码主线：

`config`
-> `tools/train.py`
-> `pointcept/engines/defaults.py`
-> `pointcept/engines/train.py`
-> `pointcept/models/build_model`
-> `DefaultSegmentorV2`
-> `PT-v3m1`
-> `pointcept/datasets/build_dataset`
-> `S3DISDataset`
-> `DefaultDataset`
-> `transform.Compose / Collect`

当前主线含义：

- config 决定模型、loss、数据集、transform、优化器和 runtime
- trainer 只要求模型前向返回统一 `loss`
- `DefaultSegmentorV2` 负责 backbone 输出到语义分割头的封装
- PTv3 backbone 处理 `Point` 结构，不建议当前阶段直接改其主干
- 数据集主线由 `S3DISDataset -> DefaultDataset -> transform -> Collect` 构成

## Appendix B. 建议重点关注文件清单

### 配置与训练入口

- `configs/s3dis/semseg-pt-v3m1-0-base.py`
- `configs/_base_/default_runtime.py`
- `tools/train.py`
- `pointcept/engines/defaults.py`
- `pointcept/engines/train.py`

### 模型最小主线

- `pointcept/models/__init__.py`
- `pointcept/models/builder.py`
- `pointcept/models/default.py`
- `pointcept/models/point_transformer_v3/point_transformer_v3m1_base.py`
- `pointcept/models/utils/structure.py`
- `pointcept/models/losses/builder.py`
- `pointcept/models/losses/__init__.py`

### 数据最小主线

- `pointcept/datasets/__init__.py`
- `pointcept/datasets/builder.py`
- `pointcept/datasets/s3dis.py`
- `pointcept/datasets/defaults.py`
- `pointcept/datasets/transform.py`

这些文件已经足够支撑当前阶段的研究计划编写与后续 Phase 1 数据接口梳理，不需要现在扩散到更多模块。

## Appendix C. 当前明确不需要阅读的部分

当前阶段明确不需要系统阅读以下内容：

- 其他 backbone 实现，例如 `sparse_unet`、`point_transformer_v2`、`octformer`、`spvcnn`、`sgiformer`
- 实例分割、预训练、提示学习、蒸馏等与当前 PTv3 语义分割基线无直接关系的模型模块
- 各类其他数据集模块，例如 ScanNet、Waymo、NuScenes、ModelNet、ShapeNetPart
- S3DIS 六折测试脚本与其他测试脚本的细节
- 复杂 loss 具体实现细节
- 与当前主线无关的预处理脚本、可视化脚本和历史实验目录

延后阅读原则：

- 只有当某一阶段的最小实现被当前文件集合阻塞时，才允许扩大阅读范围
- 每次扩大范围都应说明新增文件为什么必要
