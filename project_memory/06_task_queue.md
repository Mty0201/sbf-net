# Task Queue

1. `axis-side` smoke 核证: 在 CUDA-enabled `ptv3` 环境中使用 `semseg-pt-v3m1-0-base-bf-edge-axis-side-train-smoke.py` 确认这条已落地主线能完整跑通 train 2 epoch + validate 1 epoch；当前 workspace 仅有一次未完成的启动日志，不能视为 smoke 已通过。
2. `axis-side` full train: 仅在 smoke 确认通过后，使用 `semseg-pt-v3m1-0-base-bf-edge-axis-side-train.py` 做正式验证。
3. `B′` / `Route A` 证据补齐: 若需要把这些平行 signed-direction 路线写成“workspace 可直接复核”，先定位并确认对应 log / output；优先级低于 `axis-side` 主线核证。
4. 历史结果同步说明: 在 memory / handoff 中保留作者已确认实验事实，同时明确哪些当前 workspace 缺少可直接复核 artifact。
5. 当前主目标: 超越 `support-only best 74.6`；`73.8` 仅作为安全线。
6. 当前不做: 不重新扫 support 参数，不重开 `dist` 主线，不扩展到 Pointcept 改写、test/export/visualization。
