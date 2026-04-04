# Log Summary: train.log

## Basic Info

- File: `/home/mty0201/Pointcept/sbf-net/outputs/semseg-pt-v3m1-0-base-bf-edge-support-shape-train/train.log`
- Size: `42.92 MB` (45006615 bytes)
- Lines: `120957`
- Time range: `2026-03-30T11:12:08` -> `2026-03-30T20:39:57`
- Runs detected: `1`
- Epoch range: `1` -> `100` (hint total `2000`)
- Iter range: `1` -> `1160` (hint total `1160`)
- Checkpoint lines detected: `314`

## Sessions

- Session 1: status `validated_with_checkpoints`, device `cuda`, smoke `False`, train records `116000`, val records `3200`, checkpoints `314`
  Start `2026-03-30T11:12:08`, end `2026-03-30T20:39:57`, save path `/home/mty/Python_Proj/for_build_seg/Pointcept/sbf-net/outputs/semantic_boundary_support_shape_train`

## Last Values

### train_result

- `loss` = `0.2712`
- `loss_edge` = `0.1285`
- `loss_ordinal` = `0.0314`
- `loss_semantic` = `0.1427`
- `loss_support` = `0.1128`
- `loss_support_cover` = `0.3792`
- `loss_support_reg` = `0.0370`
- `optimizer_steps` = `194.0000`
- `ordinal_pairs` = `209.0000`

### val (step/batch-level, NOT epoch-aggregated)

- `allAcc` = `0.9416`
- `dir_cosine` = `0.0016`
- `dir_valid_ratio` = `0.1632`
- `dist_error` = `0.0374`
- `dist_gt_valid_mean` = `0.0420`
- `mAcc` = `0.9186`
- `mIoU` = `0.8852`
- `support_cover` = `0.6168`
- `support_error` = `0.0894`
- `support_positive_ratio` = `0.1633`
- `val_loss_dir` = `0.9984`
- `val_loss_dist` = `0.1463`
- `val_loss_edge` = `0.1213`
- `val_loss_support` = `0.1213`
- `val_loss_support_cover` = `0.3832`
- `val_loss_support_reg` = `0.0447`
- `valid_ratio` = `0.1633`

### scalar (epoch-aggregated, authoritative for val_mIoU)

- `best_val_mIoU` = `0.7316`
- `val_mIoU` = `0.7085`

### train

- `Lr` = `1.000e-06`
- `loss` = `0.2471`
- `loss_edge` = `0.1249`
- `loss_ordinal` = `0.0280`
- `loss_semantic` = `0.1222`
- `loss_support` = `0.1109`
- `loss_support_cover` = `0.3815`
- `loss_support_reg` = `0.0346`
- `ordinal_pairs` = `220.0000`
- `support_cover` = `0.6185`
- `support_positive_ratio` = `0.0934`
- `valid_ratio` = `0.0934`

## Best Values

### val (step/batch-level, NOT epoch-aggregated)

- `allAcc` best `0.9626` (higher_is_better, ts `2026-03-30T19:37:47`, epoch `89`, iter `6`)
- `dir_cosine` best `0.0775` (higher_is_better, ts `2026-03-30T11:18:10`, epoch `1`, iter `25`)
- `dist_error` best `0.0314` (lower_is_better, ts `2026-03-30T16:41:39`, epoch `58`, iter `1`)
- `mAcc` best `0.9718` (higher_is_better, ts `2026-03-30T17:04:29`, epoch `62`, iter `9`)
- `mIoU` best `0.9423` (higher_is_better, ts `2026-03-30T16:53:13`, epoch `60`, iter `17`)
- `support_cover` best `0.6582` (higher_is_better, ts `2026-03-30T18:35:42`, epoch `78`, iter `6`)
- `support_error` best `0.0686` (lower_is_better, ts `2026-03-30T15:27:14`, epoch `45`, iter `6`)
- `val_loss_dir` best `0.9225` (lower_is_better, ts `2026-03-30T11:18:10`, epoch `1`, iter `25`)
- `val_loss_dist` best `0.1140` (lower_is_better, ts `2026-03-30T15:44:08`, epoch `48`, iter `21`)
- `val_loss_edge` best `0.1049` (lower_is_better, ts `2026-03-30T18:41:21`, epoch `79`, iter `6`)
- `val_loss_support` best `0.1049` (lower_is_better, ts `2026-03-30T18:41:21`, epoch `79`, iter `6`)
- `val_loss_support_cover` best `0.3418` (lower_is_better, ts `2026-03-30T18:35:42`, epoch `78`, iter `6`)
- `val_loss_support_reg` best `0.0343` (lower_is_better, ts `2026-03-30T15:27:14`, epoch `45`, iter `6`)

### train_result

- `loss` best `0.2710` (lower_is_better, ts `2026-03-30T20:11:44`, epoch `95`, iter `None`)
- `loss_edge` best `0.1283` (lower_is_better, ts `2026-03-30T20:23:00`, epoch `97`, iter `None`)
- `loss_ordinal` best `0.0314` (lower_is_better, ts `2026-03-30T20:39:57`, epoch `100`, iter `None`)
- `loss_semantic` best `0.1425` (lower_is_better, ts `2026-03-30T20:11:44`, epoch `95`, iter `None`)
- `loss_support` best `0.1125` (lower_is_better, ts `2026-03-30T20:23:00`, epoch `97`, iter `None`)
- `loss_support_cover` best `0.3782` (lower_is_better, ts `2026-03-30T20:23:00`, epoch `97`, iter `None`)
- `loss_support_reg` best `0.0365` (lower_is_better, ts `2026-03-30T11:18:11`, epoch `1`, iter `None`)

### train

- `loss` best `0.1613` (lower_is_better, ts `2026-03-30T17:48:27`, epoch `70`, iter `923`)
- `loss_edge` best `0.1065` (lower_is_better, ts `2026-03-30T18:50:15`, epoch `81`, iter `662`)
- `loss_ordinal` best `0.0150` (lower_is_better, ts `2026-03-30T20:10:19`, epoch `95`, iter `873`)
- `loss_semantic` best `0.0439` (lower_is_better, ts `2026-03-30T19:27:29`, epoch `88`, iter `191`)
- `loss_support` best `0.0954` (lower_is_better, ts `2026-03-30T20:19:07`, epoch `97`, iter `358`)
- `loss_support_cover` best `0.3171` (lower_is_better, ts `2026-03-30T20:10:58`, epoch `95`, iter `1007`)
- `loss_support_reg` best `0.0300` (lower_is_better, ts `2026-03-30T17:06:25`, epoch `63`, iter `391`)
- `support_cover` best `0.6829` (higher_is_better, ts `2026-03-30T20:10:58`, epoch `95`, iter `1007`)

### scalar (epoch-aggregated, authoritative for val_mIoU)

- `best_val_mIoU` best `0.7316` (higher_is_better, ts `2026-03-30T17:21:28`, epoch `65`, iter `None`)
- `val_mIoU` best `0.7316` (higher_is_better, ts `2026-03-30T17:21:28`, epoch `65`, iter `None`)

## Recent Changes

### train (last 5)

- `2026-03-30T20:39:52` epoch 100, iter 1156: loss=0.2390, loss_semantic=0.1153, loss_edge=0.1236, loss_support=0.1099, support_cover=0.6267, Lr=1.000e-06, loss_ordinal=0.0274, loss_support_cover=0.3733
- `2026-03-30T20:39:53` epoch 100, iter 1157: loss=0.2150, loss_semantic=0.0975, loss_edge=0.1175, loss_support=0.1051, support_cover=0.6441, Lr=1.000e-06, loss_ordinal=0.0248, loss_support_cover=0.3559
- `2026-03-30T20:39:53` epoch 100, iter 1158: loss=0.2804, loss_semantic=0.1445, loss_edge=0.1360, loss_support=0.1168, support_cover=0.6073, Lr=1.000e-06, loss_ordinal=0.0383, loss_support_cover=0.3927
- `2026-03-30T20:39:53` epoch 100, iter 1159: loss=0.2569, loss_semantic=0.1319, loss_edge=0.1250, loss_support=0.1104, support_cover=0.6266, Lr=1.000e-06, loss_ordinal=0.0292, loss_support_cover=0.3734
- `2026-03-30T20:39:53` epoch 100, iter 1160: loss=0.2471, loss_semantic=0.1222, loss_edge=0.1249, loss_support=0.1109, support_cover=0.6185, Lr=1.000e-06, loss_ordinal=0.0280, loss_support_cover=0.3815

### val (step/batch-level, NOT epoch-aggregated) (last 5)

- `2026-03-30T20:39:56` step 28: mIoU=0.5297, mAcc=0.8402, allAcc=0.8194, support_cover=0.3443, support_error=0.1819, dir_cosine=-0.0227, dist_error=0.0445, val_loss_edge=0.2221
- `2026-03-30T20:39:56` step 29: mIoU=0.3616, mAcc=0.6860, allAcc=0.7081, support_cover=0.3206, support_error=0.1518, dir_cosine=0.0269, dist_error=0.0480, val_loss_edge=0.2118
- `2026-03-30T20:39:56` step 30: mIoU=0.8085, mAcc=0.9322, allAcc=0.9044, support_cover=0.5838, support_error=0.0957, dir_cosine=-0.0184, dist_error=0.0382, val_loss_edge=0.1311
- `2026-03-30T20:39:56` step 31: mIoU=0.7795, mAcc=0.8783, allAcc=0.8623, support_cover=0.4948, support_error=0.1180, dir_cosine=-0.0032, dist_error=0.0389, val_loss_edge=0.1600
- `2026-03-30T20:39:56` step 32: mIoU=0.8852, mAcc=0.9186, allAcc=0.9416, support_cover=0.6168, support_error=0.0894, dir_cosine=0.0016, dist_error=0.0374, val_loss_edge=0.1213

### train_result (last 5)

- `2026-03-30T20:17:22` no step info: loss=0.2736, loss_semantic=0.1446, loss_edge=0.1290, loss_support=0.1131, loss_ordinal=0.0319, loss_support_cover=0.3801, loss_support_reg=0.0370, optimizer_steps=194.0000
- `2026-03-30T20:23:00` no step info: loss=0.2711, loss_semantic=0.1428, loss_edge=0.1283, loss_support=0.1125, loss_ordinal=0.0316, loss_support_cover=0.3782, loss_support_reg=0.0369, optimizer_steps=194.0000
- `2026-03-30T20:28:39` no step info: loss=0.2730, loss_semantic=0.1443, loss_edge=0.1287, loss_support=0.1129, loss_ordinal=0.0315, loss_support_cover=0.3797, loss_support_reg=0.0370, optimizer_steps=194.0000
- `2026-03-30T20:34:18` no step info: loss=0.2725, loss_semantic=0.1438, loss_edge=0.1287, loss_support=0.1128, loss_ordinal=0.0316, loss_support_cover=0.3794, loss_support_reg=0.0370, optimizer_steps=194.0000
- `2026-03-30T20:39:57` no step info: loss=0.2712, loss_semantic=0.1427, loss_edge=0.1285, loss_support=0.1128, loss_ordinal=0.0314, loss_support_cover=0.3792, loss_support_reg=0.0370, optimizer_steps=194.0000

### scalar (epoch-aggregated, authoritative for val_mIoU) (last 5)

- `2026-03-30T20:28:39` `best_val_mIoU` = `0.7316`
- `2026-03-30T20:34:18` `val_mIoU` = `0.7111`
- `2026-03-30T20:34:18` `best_val_mIoU` = `0.7316`
- `2026-03-30T20:39:57` `val_mIoU` = `0.7085`
- `2026-03-30T20:39:57` `best_val_mIoU` = `0.7316`

## Loss Trends

### train

- `loss`: first `3.5417`, last `0.2471`, delta `-3.2946`, min `0.1613`, max `3.7479`, trend `down`
- `loss_edge`: first `0.2240`, last `0.1249`, delta `-0.0991`, min `0.1065`, max `0.2424`, trend `down`
- `loss_ordinal`: first `0.0680`, last `0.0280`, delta `-0.0400`, min `0.0150`, max `0.0794`, trend `down`
- `loss_semantic`: first `3.3177`, last `0.1222`, delta `-3.1955`, min `0.0439`, max `3.5204`, trend `down`
- `loss_support`: first `0.1900`, last `0.1109`, delta `-0.0791`, min `0.0954`, max `0.2158`, trend `down`
- `loss_support_cover`: first `0.7554`, last `0.3815`, delta `-0.3739`, min `0.3171`, max `0.8655`, trend `down`
- `loss_support_reg`: first `0.0389`, last `0.0346`, delta `-0.0043`, min `0.0300`, max `0.0634`, trend `down`

### val

- `val_loss_dir`: first `0.9795`, last `0.9984`, delta `0.0189`, min `0.9225`, max `1.0763`, trend `flat`
- `val_loss_dist`: first `3.2835`, last `0.1463`, delta `-3.1372`, min `0.1140`, max `4.1632`, trend `down`
- `val_loss_edge`: first `0.1949`, last `0.1213`, delta `-0.0736`, min `0.1049`, max `0.3270`, trend `down`
- `val_loss_support`: first `0.1949`, last `0.1213`, delta `-0.0736`, min `0.1049`, max `0.3270`, trend `down`
- `val_loss_support_cover`: first `0.7868`, last `0.3832`, delta `-0.4036`, min `0.3418`, max `0.8946`, trend `down`
- `val_loss_support_reg`: first `0.0375`, last `0.0447`, delta `0.0072`, min `0.0343`, max `0.1480`, trend `up`

### train_result

- `loss`: first `1.6354`, last `0.2712`, delta `-1.3642`, min `0.2710`, max `1.6354`, trend `down`
- `loss_edge`: first `0.2098`, last `0.1285`, delta `-0.0813`, min `0.1283`, max `0.2098`, trend `down`
- `loss_ordinal`: first `0.0484`, last `0.0314`, delta `-0.0170`, min `0.0314`, max `0.0484`, trend `down`
- `loss_semantic`: first `1.4257`, last `0.1427`, delta `-1.2830`, min `0.1425`, max `1.4257`, trend `down`
- `loss_support`: first `0.1856`, last `0.1128`, delta `-0.0728`, min `0.1125`, max `0.1856`, trend `down`
- `loss_support_cover`: first `0.7452`, last `0.3792`, delta `-0.3660`, min `0.3782`, max `0.7452`, trend `down`
- `loss_support_reg`: first `0.0365`, last `0.0370`, delta `5.000e-04`, min `0.0365`, max `0.0419`, trend `flat`

## Warnings And Anomalies

- Warning line 1192: `Val/Test: [1/32] val_loss_edge: 0.1949 val_loss_support: 0.1949 val_loss_support_reg: 0.0375 val_loss_support_cover: 0.7868 val_loss_dir: 0.9795 val_loss_dist: 3.2835 support_cover: 0.2132 valid_ratio: 0.0828 support_positive_ratio: 0.0828 dir_valid_ratio: 0.0828 dist_gt_valid_mean: 0.0418 dir_cosine: 0.0205 dist_error: 0.3006 support_error: 0.0751 mIoU: 0.3660 mAcc: 0.5763 allAcc: 0.6412`
- Warning line 1193: `Val/Test: [2/32] val_loss_edge: 0.1907 val_loss_support: 0.1907 val_loss_support_reg: 0.0367 val_loss_support_cover: 0.7697 val_loss_dir: 0.9613 val_loss_dist: 2.4926 support_cover: 0.2303 valid_ratio: 0.0879 support_positive_ratio: 0.0879 dir_valid_ratio: 0.0878 dist_gt_valid_mean: 0.0425 dir_cosine: 0.0387 dist_error: 0.2364 support_error: 0.0735 mIoU: 0.3694 mAcc: 0.6102 allAcc: 0.7391`
- Warning line 1194: `Val/Test: [3/32] val_loss_edge: 0.1785 val_loss_support: 0.1785 val_loss_support_reg: 0.0375 val_loss_support_cover: 0.7048 val_loss_dir: 1.0050 val_loss_dist: 2.9862 support_cover: 0.2952 valid_ratio: 0.1246 support_positive_ratio: 0.1246 dir_valid_ratio: 0.1245 dist_gt_valid_mean: 0.0427 dir_cosine: -0.0050 dist_error: 0.2767 support_error: 0.0751 mIoU: 0.5988 mAcc: 0.7938 allAcc: 0.8518`
- Warning line 1195: `Val/Test: [4/32] val_loss_edge: 0.1799 val_loss_support: 0.1799 val_loss_support_reg: 0.0372 val_loss_support_cover: 0.7135 val_loss_dir: 0.9879 val_loss_dist: 2.5156 support_cover: 0.2865 valid_ratio: 0.1231 support_positive_ratio: 0.1231 dir_valid_ratio: 0.1231 dist_gt_valid_mean: 0.0435 dir_cosine: 0.0121 dist_error: 0.2384 support_error: 0.0743 mIoU: 0.4834 mAcc: 0.7738 allAcc: 0.7520`
- Warning line 1196: `Val/Test: [5/32] val_loss_edge: 0.1924 val_loss_support: 0.1924 val_loss_support_reg: 0.0353 val_loss_support_cover: 0.7851 val_loss_dir: 0.9456 val_loss_dist: 2.2353 support_cover: 0.2149 valid_ratio: 0.0779 support_positive_ratio: 0.0779 dir_valid_ratio: 0.0779 dist_gt_valid_mean: 0.0441 dir_cosine: 0.0544 dist_error: 0.2156 support_error: 0.0707 mIoU: 0.4211 mAcc: 0.5788 allAcc: 0.7272`
- Warning line 1197: `Val/Test: [6/32] val_loss_edge: 0.1711 val_loss_support: 0.1711 val_loss_support_reg: 0.0375 val_loss_support_cover: 0.6681 val_loss_dir: 0.9638 val_loss_dist: 2.7952 support_cover: 0.3319 valid_ratio: 0.1461 support_positive_ratio: 0.1461 dir_valid_ratio: 0.1460 dist_gt_valid_mean: 0.0413 dir_cosine: 0.0362 dist_error: 0.2613 support_error: 0.0749 mIoU: 0.6487 mAcc: 0.8942 allAcc: 0.8513`
- Warning line 1198: `Val/Test: [7/32] val_loss_edge: 0.1757 val_loss_support: 0.1757 val_loss_support_reg: 0.0391 val_loss_support_cover: 0.6830 val_loss_dir: 0.9791 val_loss_dist: 2.8482 support_cover: 0.3170 valid_ratio: 0.1433 support_positive_ratio: 0.1433 dir_valid_ratio: 0.1432 dist_gt_valid_mean: 0.0420 dir_cosine: 0.0209 dist_error: 0.2655 support_error: 0.0782 mIoU: 0.6268 mAcc: 0.8453 allAcc: 0.8253`
- Warning line 1199: `Val/Test: [8/32] val_loss_edge: 0.1723 val_loss_support: 0.1723 val_loss_support_reg: 0.0372 val_loss_support_cover: 0.6759 val_loss_dir: 0.9814 val_loss_dist: 2.8383 support_cover: 0.3241 valid_ratio: 0.1398 support_positive_ratio: 0.1398 dir_valid_ratio: 0.1397 dist_gt_valid_mean: 0.0423 dir_cosine: 0.0186 dist_error: 0.2647 support_error: 0.0743 mIoU: 0.5178 mAcc: 0.8620 allAcc: 0.8250`
- Warning line 1200: `Val/Test: [9/32] val_loss_edge: 0.1708 val_loss_support: 0.1708 val_loss_support_reg: 0.0373 val_loss_support_cover: 0.6677 val_loss_dir: 0.9794 val_loss_dist: 3.1133 support_cover: 0.3323 valid_ratio: 0.1465 support_positive_ratio: 0.1465 dir_valid_ratio: 0.1464 dist_gt_valid_mean: 0.0425 dir_cosine: 0.0206 dist_error: 0.2869 support_error: 0.0746 mIoU: 0.7610 mAcc: 0.8709 allAcc: 0.8436`
- Warning line 1201: `Val/Test: [10/32] val_loss_edge: 0.1729 val_loss_support: 0.1729 val_loss_support_reg: 0.0376 val_loss_support_cover: 0.6764 val_loss_dir: 0.9742 val_loss_dist: 2.6164 support_cover: 0.3236 valid_ratio: 0.1435 support_positive_ratio: 0.1435 dir_valid_ratio: 0.1434 dist_gt_valid_mean: 0.0415 dir_cosine: 0.0258 dist_error: 0.2466 support_error: 0.0751 mIoU: 0.5230 mAcc: 0.8350 allAcc: 0.8293`
- Warning line 1202: `Val/Test: [11/32] val_loss_edge: 0.1731 val_loss_support: 0.1731 val_loss_support_reg: 0.0360 val_loss_support_cover: 0.6857 val_loss_dir: 1.0214 val_loss_dist: 3.0295 support_cover: 0.3143 valid_ratio: 0.1414 support_positive_ratio: 0.1414 dir_valid_ratio: 0.1413 dist_gt_valid_mean: 0.0433 dir_cosine: -0.0214 dist_error: 0.2806 support_error: 0.0720 mIoU: 0.4352 mAcc: 0.5986 allAcc: 0.7971`
- Warning line 1203: `Val/Test: [12/32] val_loss_edge: 0.1755 val_loss_support: 0.1755 val_loss_support_reg: 0.0354 val_loss_support_cover: 0.7000 val_loss_dir: 0.9964 val_loss_dist: 2.7898 support_cover: 0.3000 valid_ratio: 0.1282 support_positive_ratio: 0.1282 dir_valid_ratio: 0.1281 dist_gt_valid_mean: 0.0440 dir_cosine: 0.0036 dist_error: 0.2607 support_error: 0.0709 mIoU: 0.6072 mAcc: 0.8124 allAcc: 0.8320`
- Warning line 1204: `Val/Test: [13/32] val_loss_edge: 0.1798 val_loss_support: 0.1798 val_loss_support_reg: 0.0363 val_loss_support_cover: 0.7174 val_loss_dir: 0.9930 val_loss_dist: 2.8296 support_cover: 0.2826 valid_ratio: 0.1208 support_positive_ratio: 0.1208 dir_valid_ratio: 0.1208 dist_gt_valid_mean: 0.0437 dir_cosine: 0.0070 dist_error: 0.2637 support_error: 0.0727 mIoU: 0.6438 mAcc: 0.8584 allAcc: 0.8711`
- Warning line 1205: `Val/Test: [14/32] val_loss_edge: 0.1875 val_loss_support: 0.1875 val_loss_support_reg: 0.0374 val_loss_support_cover: 0.7502 val_loss_dir: 0.9980 val_loss_dist: 2.4684 support_cover: 0.2498 valid_ratio: 0.0957 support_positive_ratio: 0.0957 dir_valid_ratio: 0.0957 dist_gt_valid_mean: 0.0416 dir_cosine: 0.0020 dist_error: 0.2341 support_error: 0.0748 mIoU: 0.5378 mAcc: 0.8058 allAcc: 0.8682`
- Warning line 1206: `Val/Test: [15/32] val_loss_edge: 0.1874 val_loss_support: 0.1874 val_loss_support_reg: 0.0354 val_loss_support_cover: 0.7599 val_loss_dir: 0.9414 val_loss_dist: 2.5635 support_cover: 0.2401 valid_ratio: 0.0920 support_positive_ratio: 0.0920 dir_valid_ratio: 0.0920 dist_gt_valid_mean: 0.0441 dir_cosine: 0.0586 dist_error: 0.2422 support_error: 0.0708 mIoU: 0.4313 mAcc: 0.6410 allAcc: 0.7611`
- Warning line 1207: `Val/Test: [16/32] val_loss_edge: 0.1802 val_loss_support: 0.1802 val_loss_support_reg: 0.0387 val_loss_support_cover: 0.7077 val_loss_dir: 0.9596 val_loss_dist: 3.0747 support_cover: 0.2923 valid_ratio: 0.1239 support_positive_ratio: 0.1239 dir_valid_ratio: 0.1238 dist_gt_valid_mean: 0.0418 dir_cosine: 0.0404 dist_error: 0.2841 support_error: 0.0774 mIoU: 0.5016 mAcc: 0.8231 allAcc: 0.8028`
- Warning line 1208: `Val/Test: [17/32] val_loss_edge: 0.1820 val_loss_support: 0.1820 val_loss_support_reg: 0.0402 val_loss_support_cover: 0.7092 val_loss_dir: 0.9270 val_loss_dist: 2.8132 support_cover: 0.2908 valid_ratio: 0.1164 support_positive_ratio: 0.1164 dir_valid_ratio: 0.1164 dist_gt_valid_mean: 0.0431 dir_cosine: 0.0730 dist_error: 0.2629 support_error: 0.0803 mIoU: 0.6750 mAcc: 0.8983 allAcc: 0.8558`
- Warning line 1209: `Val/Test: [18/32] val_loss_edge: 0.1691 val_loss_support: 0.1691 val_loss_support_reg: 0.0360 val_loss_support_cover: 0.6653 val_loss_dir: 1.0319 val_loss_dist: 3.1698 support_cover: 0.3347 valid_ratio: 0.1520 support_positive_ratio: 0.1520 dir_valid_ratio: 0.1520 dist_gt_valid_mean: 0.0427 dir_cosine: -0.0319 dist_error: 0.2915 support_error: 0.0721 mIoU: 0.6203 mAcc: 0.8252 allAcc: 0.8377`
- Warning line 1210: `Val/Test: [19/32] val_loss_edge: 0.1827 val_loss_support: 0.1827 val_loss_support_reg: 0.0394 val_loss_support_cover: 0.7167 val_loss_dir: 0.9759 val_loss_dist: 3.1582 support_cover: 0.2833 valid_ratio: 0.1239 support_positive_ratio: 0.1239 dir_valid_ratio: 0.1239 dist_gt_valid_mean: 0.0418 dir_cosine: 0.0241 dist_error: 0.2905 support_error: 0.0787 mIoU: 0.4390 mAcc: 0.7300 allAcc: 0.7302`
- Warning line 1211: `Val/Test: [20/32] val_loss_edge: 0.1942 val_loss_support: 0.1942 val_loss_support_reg: 0.0376 val_loss_support_cover: 0.7830 val_loss_dir: 0.9939 val_loss_dist: 2.8360 support_cover: 0.2170 valid_ratio: 0.0807 support_positive_ratio: 0.0807 dir_valid_ratio: 0.0806 dist_gt_valid_mean: 0.0425 dir_cosine: 0.0061 dist_error: 0.2645 support_error: 0.0753 mIoU: 0.4787 mAcc: 0.6777 allAcc: 0.7918`

## Auto Questions

- None.
