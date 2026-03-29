# Task Queue

1. `axis-side` smoke 确认: 在 CUDA-enabled `ptv3` 环境中使用 `semseg-pt-v3m1-0-base-bf-edge-axis-side-train-smoke.py` 确认该路线能完整跑通 train 2 epoch + validate 1 epoch。
2. `axis-side` full train: 仅在 smoke 确认通过后，使用 `semseg-pt-v3m1-0-base-bf-edge-axis-side-train.py` 做正式验证。
3. `B′` 结果核证: 若需要把 `B′ full train` 写成已完成，先定位并确认对应 log / output；当前 workspace 中尚无已确认的 `≈72.8` 证据。
4. `Route A` full train 仍是可选后续路线，但当前 workspace 的新增代码状态优先要求先核清 `axis-side`。
5. 当前主目标: 超越 `support-only best 74.6`；`73.8` 仅作为安全线。
6. 当前不做: 不重新扫 support 参数，不重开 `dist` 主线，不扩展到 Pointcept 改写、test/export/visualization。
