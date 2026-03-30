# Log Summary: train.log

## Basic Info

- File: `/home/mty0201/Pointcept/semantic-boundary-field/outputs/semantic_boundary_axis_side_train_smoke/train.log`
- Size: `9.53 KB` (9755 bytes)
- Lines: `142`
- Time range: `2026-03-29T19:11:01` -> `2026-03-30T01:08:03`
- Runs detected: `4`
- Epoch range: `1` -> `1` (hint total `2`)
- Iter range: `1` -> `2` (hint total `2`)
- Checkpoint lines detected: `4`

## Sessions

- Session 1: status `startup_only`, device `cpu`, smoke `True`, train records `0`, val records `0`, checkpoints `0`
  Start `2026-03-29T19:11:01`, end `2026-03-29T19:11:01`, save path `/home/mty0201/Pointcept/semantic-boundary-field/outputs/semantic_boundary_axis_side_train_smoke`
- Session 2: status `startup_only`, device `cpu`, smoke `True`, train records `0`, val records `0`, checkpoints `0`
  Start `2026-03-29T21:50:02`, end `2026-03-29T21:50:03`, save path `/home/mty0201/Pointcept/semantic-boundary-field/outputs/semantic_boundary_axis_side_train_smoke`
- Session 3: status `startup_only`, device `cpu`, smoke `True`, train records `0`, val records `0`, checkpoints `0`
  Start `2026-03-29T21:51:32`, end `2026-03-29T21:51:32`, save path `/home/mty0201/Pointcept/semantic-boundary-field/outputs/semantic_boundary_axis_side_train_smoke`
- Session 4: status `validated_with_checkpoints`, device `cuda`, smoke `True`, train records `2`, val records `1`, checkpoints `4`
  Start `2026-03-30T01:07:52`, end `2026-03-30T01:08:03`, save path `/home/mty0201/Pointcept/semantic-boundary-field/outputs/semantic_boundary_axis_side_train_smoke`

## Last Values

### train_result

- `axis_cosine` = `0.4850`
- `loss` = `4.5781`
- `loss_axis` = `0.5225`
- `loss_edge` = `1.4059`
- `loss_semantic` = `3.1722`
- `loss_side` = `0.7127`
- `loss_support` = `0.1707`
- `optimizer_steps` = `1.0000`
- `side_accuracy` = `0.4910`

### val

- `allAcc` = `0.1001`
- `axis_cosine` = `0.4411`
- `dir_cosine` = `0.0143`
- `mAcc` = `0.1576`
- `mIoU` = `0.0378`
- `side_accuracy` = `0.4602`
- `support_cover` = `0.3061`
- `val_loss_axis` = `0.5654`
- `val_loss_edge` = `1.4745`
- `val_loss_side` = `0.7295`
- `val_loss_support` = `0.1796`

### scalar

- `best_val_mIoU` = `0.0378`
- `val_mIoU` = `0.0378`

### train

- `Lr` = `6.000e-06`
- `axis_cosine` = `0.4827`
- `axis_valid_ratio` = `0.1654`
- `dir_cosine` = `-0.0045`
- `loss` = `4.6192`
- `loss_axis` = `0.5237`
- `loss_edge` = `1.4105`
- `loss_semantic` = `3.2087`
- `loss_side` = `0.7138`
- `loss_support` = `0.1730`
- `loss_support_cover` = `0.6570`
- `loss_support_reg` = `0.0416`
- `side_accuracy` = `0.4982`
- `support_positive_ratio` = `0.1654`
- `valid_ratio` = `0.1654`

## Best Values

### val

- `allAcc` best `0.1001` (higher_is_better, ts `2026-03-30T01:08:00`, epoch `1`, iter `1`)
- `axis_cosine` best `0.4411` (higher_is_better, ts `2026-03-30T01:08:00`, epoch `1`, iter `1`)
- `dir_cosine` best `0.0143` (higher_is_better, ts `2026-03-30T01:08:00`, epoch `1`, iter `1`)
- `mAcc` best `0.1576` (higher_is_better, ts `2026-03-30T01:08:00`, epoch `1`, iter `1`)
- `mIoU` best `0.0378` (higher_is_better, ts `2026-03-30T01:08:00`, epoch `1`, iter `1`)
- `side_accuracy` best `0.4602` (higher_is_better, ts `2026-03-30T01:08:00`, epoch `1`, iter `1`)
- `support_cover` best `0.3061` (higher_is_better, ts `2026-03-30T01:08:00`, epoch `1`, iter `1`)
- `val_loss_axis` best `0.5654` (lower_is_better, ts `2026-03-30T01:08:00`, epoch `1`, iter `1`)
- `val_loss_edge` best `1.4745` (lower_is_better, ts `2026-03-30T01:08:00`, epoch `1`, iter `1`)
- `val_loss_side` best `0.7295` (lower_is_better, ts `2026-03-30T01:08:00`, epoch `1`, iter `1`)
- `val_loss_support` best `0.1796` (lower_is_better, ts `2026-03-30T01:08:00`, epoch `1`, iter `1`)

### train_result

- `axis_cosine` best `0.4850` (higher_is_better, ts `2026-03-30T01:08:03`, epoch `1`, iter `None`)
- `loss` best `4.5781` (lower_is_better, ts `2026-03-30T01:08:03`, epoch `1`, iter `None`)
- `loss_axis` best `0.5225` (lower_is_better, ts `2026-03-30T01:08:03`, epoch `1`, iter `None`)
- `loss_edge` best `1.4059` (lower_is_better, ts `2026-03-30T01:08:03`, epoch `1`, iter `None`)
- `loss_semantic` best `3.1722` (lower_is_better, ts `2026-03-30T01:08:03`, epoch `1`, iter `None`)
- `loss_side` best `0.7127` (lower_is_better, ts `2026-03-30T01:08:03`, epoch `1`, iter `None`)
- `loss_support` best `0.1707` (lower_is_better, ts `2026-03-30T01:08:03`, epoch `1`, iter `None`)
- `side_accuracy` best `0.4910` (higher_is_better, ts `2026-03-30T01:08:03`, epoch `1`, iter `None`)

### train

- `axis_cosine` best `0.4874` (higher_is_better, ts `2026-03-30T01:07:59`, epoch `1`, iter `1`)
- `dir_cosine` best `0.0162` (higher_is_better, ts `2026-03-30T01:07:59`, epoch `1`, iter `1`)
- `loss` best `4.5370` (lower_is_better, ts `2026-03-30T01:07:59`, epoch `1`, iter `1`)
- `loss_axis` best `0.5214` (lower_is_better, ts `2026-03-30T01:07:59`, epoch `1`, iter `1`)
- `loss_edge` best `1.4014` (lower_is_better, ts `2026-03-30T01:07:59`, epoch `1`, iter `1`)
- `loss_semantic` best `3.1357` (lower_is_better, ts `2026-03-30T01:07:59`, epoch `1`, iter `1`)
- `loss_side` best `0.7116` (lower_is_better, ts `2026-03-30T01:07:59`, epoch `1`, iter `1`)
- `loss_support` best `0.1683` (lower_is_better, ts `2026-03-30T01:07:59`, epoch `1`, iter `1`)
- `loss_support_cover` best `0.6385` (lower_is_better, ts `2026-03-30T01:07:59`, epoch `1`, iter `1`)
- `loss_support_reg` best `0.0406` (lower_is_better, ts `2026-03-30T01:07:59`, epoch `1`, iter `1`)
- `side_accuracy` best `0.4982` (higher_is_better, ts `2026-03-30T01:08:00`, epoch `1`, iter `2`)

### scalar

- `best_val_mIoU` best `0.0378` (higher_is_better, ts `2026-03-30T01:08:03`, epoch `1`, iter `None`)
- `val_mIoU` best `0.0378` (higher_is_better, ts `2026-03-30T01:08:03`, epoch `1`, iter `None`)

## Recent Changes

### train (last 2)

- `2026-03-30T01:07:59` epoch 1, iter 1: loss=4.5370, loss_semantic=3.1357, loss_edge=1.4014, loss_support=0.1683, loss_axis=0.5214, loss_side=0.7116, axis_cosine=0.4874, side_accuracy=0.4837
- `2026-03-30T01:08:00` epoch 1, iter 2: loss=4.6192, loss_semantic=3.2087, loss_edge=1.4105, loss_support=0.1730, loss_axis=0.5237, loss_side=0.7138, axis_cosine=0.4827, side_accuracy=0.4982

### val (last 1)

- `2026-03-30T01:08:00` step 1: mIoU=0.0378, mAcc=0.1576, allAcc=0.1001, support_cover=0.3061, axis_cosine=0.4411, side_accuracy=0.4602, dir_cosine=0.0143, val_loss_edge=1.4745

### train_result (last 1)

- `2026-03-30T01:08:03` no step info: loss=4.5781, loss_semantic=3.1722, loss_edge=1.4059, loss_support=0.1707, loss_axis=0.5225, loss_side=0.7127, axis_cosine=0.4850, side_accuracy=0.4910

### scalar (last 2)

- `2026-03-30T01:08:03` `val_mIoU` = `0.0378`
- `2026-03-30T01:08:03` `best_val_mIoU` = `0.0378`

## Loss Trends

### train

- `loss`: first `4.5370`, last `4.6192`, delta `0.0822`, min `4.5370`, max `4.6192`, trend `flat`
- `loss_axis`: first `0.5214`, last `0.5237`, delta `0.0023`, min `0.5214`, max `0.5237`, trend `flat`
- `loss_edge`: first `1.4014`, last `1.4105`, delta `0.0091`, min `1.4014`, max `1.4105`, trend `flat`
- `loss_semantic`: first `3.1357`, last `3.2087`, delta `0.0730`, min `3.1357`, max `3.2087`, trend `up`
- `loss_side`: first `0.7116`, last `0.7138`, delta `0.0022`, min `0.7116`, max `0.7138`, trend `flat`
- `loss_support`: first `0.1683`, last `0.1730`, delta `0.0047`, min `0.1683`, max `0.1730`, trend `up`
- `loss_support_cover`: first `0.6385`, last `0.6570`, delta `0.0185`, min `0.6385`, max `0.6570`, trend `up`
- `loss_support_reg`: first `0.0406`, last `0.0416`, delta `0.0010`, min `0.0406`, max `0.0416`, trend `up`

### val

- `val_loss_axis`: first `0.5654`, last `0.5654`, delta `0.0000`, min `0.5654`, max `0.5654`, trend `flat`
- `val_loss_edge`: first `1.4745`, last `1.4745`, delta `0.0000`, min `1.4745`, max `1.4745`, trend `flat`
- `val_loss_side`: first `0.7295`, last `0.7295`, delta `0.0000`, min `0.7295`, max `0.7295`, trend `flat`
- `val_loss_support`: first `0.1796`, last `0.1796`, delta `0.0000`, min `0.1796`, max `0.1796`, trend `flat`

### train_result

- `loss`: first `4.5781`, last `4.5781`, delta `0.0000`, min `4.5781`, max `4.5781`, trend `flat`
- `loss_axis`: first `0.5225`, last `0.5225`, delta `0.0000`, min `0.5225`, max `0.5225`, trend `flat`
- `loss_edge`: first `1.4059`, last `1.4059`, delta `0.0000`, min `1.4059`, max `1.4059`, trend `flat`
- `loss_semantic`: first `3.1722`, last `3.1722`, delta `0.0000`, min `3.1722`, max `3.1722`, trend `flat`
- `loss_side`: first `0.7127`, last `0.7127`, delta `0.0000`, min `0.7127`, max `0.7127`, trend `flat`
- `loss_support`: first `0.1707`, last `0.1707`, delta `0.0000`, min `0.1707`, max `0.1707`, trend `flat`

## Warnings And Anomalies

- Anomaly line 91: `Detected 4 training starts in one log file.`

## Auto Questions

- Why were there 4 startup attempts in the same log, and which session should be treated as authoritative?
- Is the latest val_mIoU 0.0378 expected for this run, or does it point to a config / data / label mismatch?
