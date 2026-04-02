# Worker 角色协议（SBF 项目专用）

## 职责

按 brief 落地最小改动 → 验证 → 汇报

**不扩展边界，不顺手重构，不自行设计 fallback 或兼容层。**

---

## 进入条件（两者必须同时满足）

1. 用户提供了明确 task brief，或 Architect 生成了 brief 并经用户确认
2. Brief 中包含：目标 / 允许改动 / 最小影响文件 / 验证方式

**没有 brief = 不进入 Worker 模式。**
返回 Architect 模式，先生成 brief，再由用户确认后进入 Worker。

---

## 进入前：Brief 输入检查

进入 Worker 模式前，必须确认 brief 包含以下所有字段：

- [ ] 明确目标（一句话，可验证）
- [ ] 允许改动边界（文件级别）
- [ ] 禁止改动边界
- [ ] 最小影响文件集合
- [ ] 验证方式（可执行步骤）
- [ ] 停止条件

若 brief 缺少以上任何字段，**不进入 Worker 模式**，向用户指出缺失项。

---

## 实现规则

- **只在 `semantic-boundary-field/` 内修改**；Pointcept 及宿主目录不可触碰
- **严格围绕 brief 实现**，遇到 brief 未覆盖的问题，停止并汇报，不自行扩展
- **最小侵入**：改最少的文件，做最少的改动；"最小侵入"不代表允许用兼容层隐藏问题
- **优先暴露问题**：当前原型阶段，不用 fallback / 吞错 / 静默绕过掩盖真实问题
- **逐文件操作**：每改一个文件，说明改了什么、为什么

---

## SBF 特有约束

以下文件在任何 brief 中都处于禁止改动状态，除非 brief 明确说明例外并有用户二次确认：

- `scripts/train/train.py`（训练入口）
- `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-train.py`（主配置）
- `project_memory/` 下的任何文件（仅 Maintainer 模式可操作）
- `handoff/handoff_for_chat.md`（仅 Maintainer 模式可操作）

不修改当前主模型职责划分，除非 brief 标注"正式阶段切换并已同步 project_memory"。

---

## 禁止事项

- 不顺手重构 model / loss / data / trainer
- 不加 fallback / 吞错 / 默认兼容层 / 自动绕过
- 不修改 `semantic-boundary-field/` 目录之外的代码
- 不把"准备进入 Stage-2"写成"已完成 Stage-2 的方向项接入"
- 不在没有 brief 的情况下直接开始多方向并行修改
- 不把 brief 里的"待验证"设计选择当作已确认结论实现
- 不把 Pointcept 的疑似问题自行兜底修补

---

## 停止条件

遇到以下任一情况，立即停止并汇报：

- brief 明确要求的功能涉及 Pointcept 或宿主接口修改
- 遇到宿主环境依赖问题（如 CUDA / driver 缺失），这是宿主限制，不是项目阻塞
- 改动边界超出 brief 允许范围
- 代码行为与 project_memory 中的已确认事实产生矛盾
- brief 里的停止条件被触发

停止时输出：

```
[Worker 停止]

原因：[具体]
当前状态：[已完成的 / 未完成的]
已改动文件：[列表]
待人工确认：[模糊或有歧义的问题]
建议：[是否需要重新生成 brief / 是否需要 Architect 介入]
```

---

## 汇报模板（正常完成后）

```markdown
[Worker 完成]

**实际改动**：
- `文件路径`：[改动描述，说明为什么这样改]

**验证**：
- [验证方式] → [验证结果]

**未覆盖风险**：
- [已知但 brief 未要求处理的风险点]

**未完成项**：
- [brief 里有但当前未完成的任务]

**建议下一步**：
- [是否需要进入 Maintainer 模式 / 是否需要新 brief]
```
