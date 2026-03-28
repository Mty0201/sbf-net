# Task Queue

1. B′ full train（D1-O0）: 使用 `stage2-bprime-train.py`，验证 support-weighted direction/distance 监督是否改善 v2 基线。
2. Route A full train（D1-O1）: 使用 `route-a-train.py`，验证显式 basin coherence 是否在 B′ 基础上有增量。
3. B′ vs A 对照: B′ 先于 Route A 完成 full train；两者构成 D1-O0 vs D1-O1 实验对照。
4. 当前主目标: 超越 `support-only best 74.6`；`73.8` 仅作为安全线。
5. 当前不做: 不重新扫 support 参数，不重开 `dist` 主线，不扩展到 Pointcept 改写、test/export/visualization。
