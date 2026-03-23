# Project Overview

- 项目目标: 在 Pointcept PTv3 backbone 上训练带 semantic boundary field 监督的点云语义分割模型。
- 当前主线任务: 维护并验证 stage-1 双任务训练主线，联合学习 semantic + direction + distance + support。
- 非目标: 不重写 Pointcept 框架。
- 非目标: 不把 semantic-only calibration 作为主任务。
- 非目标: 不扩展 test pipeline、result export、visualization export、distributed training。
