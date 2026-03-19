# Pointcept Boundary

## 1. 文档目的

本文档用于说明本独立项目仓库与 Pointcept 之间的工程边界。

目标是确保：

- 本项目以独立 Git 仓库发布
- Pointcept 作为前置依赖存在
- 本项目不演变成 Pointcept 的复制粘贴版或长期私有分叉

## 2. Pointcept 承担什么

在当前规划下，Pointcept 主要承担：

- PTv3 语义分割基线
- 已有训练与配置体系
- 已有基础数据读取与 transform 主线
- 已有可复用的模型主干能力

## 3. 本项目仓库承担什么

本项目仓库主要承担：

- 任务定义
- 项目文档
- 项目级配置
- 项目级脚本
- 后续的任务增量实现
- 可选的最小补丁说明

## 4. 边界原则

后续开发必须优先遵守以下边界原则：

- 能外置到本项目仓库的，不放进 Pointcept
- 能通过注册或配置接入的，不修改 Pointcept 源码
- 只有在 Pointcept 现有扩展点不足时，才允许最小修改 Pointcept
- 即使必须修改 Pointcept，也应保持为少量、可审计、可回退的补丁

## 5. 侵入等级

### A 类：完全可外置

- 项目文档
- 数据格式定义
- 实验配置
- 运行脚本
- 可视化脚本
- 预处理脚本

### B 类：通过注册或配置可接入

- 新数据集包装
- 新 transform
- 新模型壳层
- 新 head
- 新 loss 封装
- 新 hook

### C 类：可能需要最小修改 Pointcept

- 数据字段白名单
- 点级索引同步字段
- 少量外部项目导入或注册入口
- 极少量配置加载兼容点

### D 类：当前明确禁止

- 复制整个 Pointcept
- 长期维护私有 Pointcept 分叉
- 重写 PTv3 backbone
- 第一阶段直接改 trainer 主循环
- 发散到其他 backbone 或其他数据集

## 6. 当前结论

当前推荐路线是：

- 先保持 Pointcept 不动
- 先在本项目仓库内完成文档、结构与边界建设
- 后续每轮开发都先判断新增内容属于 A/B/C/D 哪一类
- 只有在 A/B 无法满足需求时，才讨论 C 类最小补丁

## 7. 外部 index_valid_keys 注入路径

当前已验证的外部方案原理如下：

- 在独立仓库的 dataset 扩展中，把 `edge.npy` 读取进 `data_dict`
- 在进入任何会触发 `index_operator` 的 Pointcept transform 之前
- 通过独立仓库的前置 transform 向 `data_dict["index_valid_keys"]` 追加 `"edge"`

该方案的目标是：

- 不修改 Pointcept 源码
- 仍然利用 Pointcept 现有 `index_operator` 机制
- 让 `edge` 作为逐点字段和 `coord / segment` 一样参与索引同步

如果该方案后续失败，当前最小 patch 点应优先定位在：

- `transform/index_operator`

当前仍保持以下边界不变：

- 尚未修改 Pointcept
- patch 只作为最小兜底选项记录
