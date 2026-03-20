"""Minimal project-local trainer for stage-1 semantic boundary training."""

from __future__ import annotations

import os
import time
import math
import random
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import project.datasets  # noqa: F401
import project.models  # noqa: F401
import project.transforms  # noqa: F401
from pointcept.datasets import build_dataset
from pointcept.datasets.utils import point_collate_fn
from pointcept.models import build_model
from project.evaluator import build_evaluator
from project.losses import build_loss
from project.utils import AverageMeter, create_logger


class IdentityPointBackbone(nn.Module):
    """CPU-safe backbone fallback for smoke tests when CUDA is unavailable."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Linear(in_channels, out_channels)

    def forward(self, point):
        point.feat = self.proj(point.feat)
        return point


class SemanticBoundaryTrainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trainer_cfg = cfg["trainer"]
        self.data_cfg = cfg["data"]
        self.optimizer_cfg = cfg["optimizer"]
        self.smoke_mode = "actual_backbone_cuda"
        self.resume = bool(cfg.get("resume", False))
        self.weight = cfg.get("weight")
        self.seed = cfg.get("seed")
        self.work_dir = cfg["work_dir"]
        self.runtime_cfg = cfg.get("runtime", {})
        self.scheduler_cfg = cfg.get("scheduler")
        self.param_dicts = cfg.get("param_dicts")
        self.best_val_miou = float("-inf")
        self.start_epoch = 1
        self.total_epoch = int(
            self.trainer_cfg.get("total_epoch", self.trainer_cfg.get("epochs", 1))
        )
        self.max_epoch = int(self.trainer_cfg.get("eval_epoch", self.total_epoch))
        if self.total_epoch % self.max_epoch != 0:
            raise ValueError("total_epoch must be divisible by eval_epoch.")
        self.train_loop = self.total_epoch // self.max_epoch
        self.log_freq = int(self.runtime_cfg.get("log_freq", 1))
        self.val_log_freq = int(self.runtime_cfg.get("val_log_freq", 10))
        self.save_freq = self.runtime_cfg.get("save_freq")
        self.grad_accum_steps = int(self.runtime_cfg.get("grad_accum_steps", 1))
        self.train_batch_size = int(
            self.data_cfg.get("train_batch_size", self.trainer_cfg["batch_size"])
        )
        self.val_batch_size = int(self.data_cfg.get("val_batch_size", 1))
        self.model_dir = os.path.join(self.work_dir, "model")

        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        self._set_seed()
        self.logger = create_logger(
            "sbf_net.trainer", os.path.join(self.work_dir, "train.log")
        )
        self.logger.info("=> Loading config ...")
        self.logger.info(f"Save path: {self.work_dir}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Seed: {self.seed}")
        self.logger.info(f"Resume: {self.resume}")
        self.logger.info(f"Weight: {self.weight}")
        self.logger.info(f"Train batch size: {self.train_batch_size}")
        self.logger.info(f"Val batch size: {self.val_batch_size}")
        self.logger.info(f"Grad accumulation steps: {self.grad_accum_steps}")
        self.logger.info(
            f"Effective batch size: {self.train_batch_size * self.grad_accum_steps}"
        )

        self.logger.info("=> Building train dataset & dataloader ...")
        self.train_dataset = build_dataset(cfg["data"]["train"])
        self.train_loader = self._build_dataloader(self.train_dataset, training=True)
        self.logger.info("=> Building val dataset & dataloader ...")
        self.val_dataset = build_dataset(cfg["data"]["val"])
        self.val_loader = self._build_dataloader(self.val_dataset, training=False)
        self.base_steps_per_epoch = self._compute_steps_per_epoch()
        self.steps_per_epoch = self.base_steps_per_epoch * self.train_loop
        self.optimizer_steps_per_epoch = math.ceil(
            self.steps_per_epoch / self.grad_accum_steps
        )
        self.total_train_iters = self.max_epoch * self.steps_per_epoch

        self.logger.info("=> Building model ...")
        self.model = build_model(cfg["model"]).to(self.device)
        self.logger.info("=> Building loss / evaluator ...")
        self.loss_fn = build_loss(cfg.get("loss")).to(self.device)
        self.evaluator = build_evaluator(cfg.get("evaluator"))
        self.class_names = cfg["data"].get("names")
        self.logger.info("=> Building optimizer / scheduler ...")
        self.optimizer = self._build_optimizer()
        self.logger.info(f"Optimizer: {self.optimizer.__class__.__name__}")
        for idx, group in enumerate(self.optimizer.param_groups):
            self.logger.info(f"LR groups: group_{idx} lr: {group['lr']}")
        self.scheduler = self._build_scheduler()
        self._cpu_backbone_ready = False
        self._prepare_cpu_backbone_for_runtime()
        self.logger.info("=> Loading checkpoint / weight if needed ...")
        self._load_checkpoint_or_weight()

    def _set_seed(self):
        if self.seed is None:
            return
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _worker_init_fn(self, worker_id: int):
        if self.seed is None:
            return
        worker_seed = self.seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    def _build_dataloader(self, dataset, training: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.train_batch_size if training else self.val_batch_size,
            shuffle=training,
            num_workers=self.trainer_cfg["num_workers"],
            collate_fn=partial(point_collate_fn, mix_prob=0),
            pin_memory=torch.cuda.is_available(),
            drop_last=training,
            persistent_workers=self.trainer_cfg["num_workers"] > 0,
            worker_init_fn=self._worker_init_fn if self.seed is not None else None,
        )

    def _compute_steps_per_epoch(self) -> int:
        if self.trainer_cfg.get("max_train_batches") is not None:
            return min(len(self.train_loader), int(self.trainer_cfg["max_train_batches"]))
        return len(self.train_loader)

    def _build_optimizer(self):
        optimizer_type = self.optimizer_cfg["type"]
        kwargs = {
            key: value
            for key, value in self.optimizer_cfg.items()
            if key not in {"type", "param_dicts"}
        }
        params = self.model.parameters()
        if self.param_dicts:
            base_lr = kwargs["lr"]
            params = [dict(params=[], lr=base_lr)]
            for group_cfg in self.param_dicts:
                params.append(
                    dict(
                        params=[],
                        lr=group_cfg["lr"],
                    )
                )
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                matched = False
                for idx, group_cfg in enumerate(self.param_dicts):
                    if group_cfg["keyword"] in name:
                        params[idx + 1]["params"].append(param)
                        matched = True
                        break
                if not matched:
                    params[0]["params"].append(param)
            kwargs["params"] = params
        else:
            kwargs["params"] = params
        if optimizer_type == "Adam":
            return torch.optim.Adam(**kwargs)
        if optimizer_type == "AdamW":
            return torch.optim.AdamW(**kwargs)
        if optimizer_type == "SGD":
            return torch.optim.SGD(**kwargs)
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    def _build_scheduler(self):
        if self.scheduler_cfg is None:
            return None
        scheduler_type = self.scheduler_cfg["type"]
        kwargs = {
            key: value for key, value in self.scheduler_cfg.items() if key != "type"
        }
        if scheduler_type == "OneCycleLR":
            kwargs.setdefault("max_lr", self.optimizer_cfg["lr"])
            kwargs["total_steps"] = self.max_epoch * self.optimizer_steps_per_epoch
            return torch.optim.lr_scheduler.OneCycleLR(self.optimizer, **kwargs)
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    def _move_batch_to_device(self, batch: dict) -> dict:
        moved = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(self.device)
            else:
                moved[key] = value
        return moved

    def _ensure_cpu_backbone(self, batch: dict):
        if torch.cuda.is_available():
            return
        if not self.trainer_cfg.get("cpu_fallback_shell_backbone", False):
            return
        if self._cpu_backbone_ready:
            return

        self.model.backbone = IdentityPointBackbone(
            in_channels=batch["feat"].shape[1],
            out_channels=self.model.semantic_head.proj.in_features,
        ).to(self.device)
        self._cpu_backbone_ready = True
        self.smoke_mode = "shell_only_no_cuda"
        self.logger.info("Using CPU shell backbone fallback for this run.")

    def _prepare_cpu_backbone_for_runtime(self):
        if torch.cuda.is_available():
            return
        if not self.trainer_cfg.get("cpu_fallback_shell_backbone", False):
            return
        if self._cpu_backbone_ready:
            return
        in_channels = int(self.cfg["model"]["backbone"]["in_channels"])
        self.model.backbone = IdentityPointBackbone(
            in_channels=in_channels,
            out_channels=self.model.semantic_head.proj.in_features,
        ).to(self.device)
        self._cpu_backbone_ready = True
        self.smoke_mode = "shell_only_no_cuda"
        self.logger.info("Prepared CPU shell backbone before checkpoint loading.")

    @staticmethod
    def _forward_input_from_batch(batch: dict) -> dict:
        return {
            "coord": batch["coord"],
            "grid_coord": batch["grid_coord"],
            "feat": batch["feat"],
            "offset": batch["offset"],
        }

    @staticmethod
    def _detach_scalar_dict(data: dict[str, torch.Tensor]) -> dict[str, float]:
        result = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                result[key] = float(value.detach().cpu())
        return result

    @staticmethod
    def _build_loss_inputs(output: dict, batch: dict) -> dict:
        kwargs = dict(seg_logits=output["seg_logits"], segment=batch["segment"])
        if "edge_pred" in output and "edge" in batch:
            kwargs["edge_pred"] = output["edge_pred"]
            kwargs["edge"] = batch["edge"]
        return kwargs

    @staticmethod
    def _build_eval_inputs(output: dict, batch: dict) -> dict:
        kwargs = dict(seg_logits=output["seg_logits"], segment=batch["segment"])
        if "edge_pred" in output and "edge" in batch:
            kwargs["edge_pred"] = output["edge_pred"]
            kwargs["edge"] = batch["edge"]
        return kwargs

    @staticmethod
    def _compute_per_class_from_stats(
        intersection: torch.Tensor,
        union: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid_iou = union > 0
        valid_acc = target > 0
        iou_class = torch.zeros_like(intersection)
        acc_class = torch.zeros_like(intersection)
        iou_class[valid_iou] = intersection[valid_iou] / union[valid_iou]
        acc_class[valid_acc] = intersection[valid_acc] / target[valid_acc]
        return iou_class, acc_class

    @staticmethod
    def _format_seconds(seconds: float) -> str:
        seconds = max(int(seconds), 0)
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    @staticmethod
    def _extract_model_state_dict(checkpoint: dict) -> dict:
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            normalized = {}
            for key, value in state_dict.items():
                normalized[key[7:] if key.startswith("module.") else key] = value
            return normalized
        raise KeyError("No model weights found in checkpoint.")

    def _load_checkpoint_or_weight(self):
        weight_path = self.weight
        if self.resume and weight_path is None:
            candidate = os.path.join(self.model_dir, "model_last.pth")
            if os.path.isfile(candidate):
                weight_path = candidate
        if weight_path is None:
            self.logger.info("No checkpoint or weight provided.")
            self.logger.info(f"Start epoch: {self.start_epoch}")
            return
        if not os.path.isfile(weight_path):
            raise FileNotFoundError(f"Checkpoint/weight not found: {weight_path}")

        checkpoint = torch.load(weight_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(self._extract_model_state_dict(checkpoint), strict=True)

        if self.resume:
            self.logger.info(f"Resuming from checkpoint: {weight_path}")
            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            elif "optimizer" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            if self.scheduler is not None:
                if "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
                    self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                elif "scheduler" in checkpoint and checkpoint["scheduler"] is not None:
                    self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.best_val_miou = float(checkpoint.get("best_val_miou", checkpoint.get("best_metric_value", float("-inf"))))
            self.start_epoch = int(checkpoint.get("epoch", 0)) + 1
            self.logger.info(f"Loaded optimizer/scheduler state. Start epoch: {self.start_epoch}")
            self.logger.info(f"Loaded best_val_mIoU: {self.best_val_miou:.6f}")
        else:
            self.logger.info(f"Loaded weight only: {weight_path}")
            self.logger.info(f"Start epoch: {self.start_epoch}")

    def _current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        loss_meters = None
        data_time_meter = AverageMeter()
        batch_time_meter = AverageMeter()
        max_batches = self.trainer_cfg.get("max_train_batches")
        iter_start = time.perf_counter()
        accum_counter = 0
        optimizer_steps = 0
        self.optimizer.zero_grad(set_to_none=True)
        processed_iter = 0
        global_batch_time_meter = AverageMeter()
        for loop_idx in range(self.train_loop):
            for batch_idx, batch in enumerate(self.train_loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                data_time = time.perf_counter() - iter_start
                data_time_meter.update(data_time)
                batch = self._move_batch_to_device(batch)
                self._ensure_cpu_backbone(batch)

                output = self.model(self._forward_input_from_batch(batch))
                loss_dict = self.loss_fn(**self._build_loss_inputs(output, batch))
                detached = self._detach_scalar_dict(loss_dict)
                if loss_meters is None:
                    loss_meters = {key: AverageMeter() for key in detached.keys()}
                scaled_loss = loss_dict["loss"] / self.grad_accum_steps
                scaled_loss.backward()
                accum_counter += 1
                processed_iter += 1
                should_step = (
                    accum_counter >= self.grad_accum_steps
                    or processed_iter >= self.steps_per_epoch
                )
                if should_step:
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    accum_counter = 0
                    optimizer_steps += 1

                for key, value in detached.items():
                    loss_meters[key].update(value)

                batch_time = time.perf_counter() - iter_start
                batch_time_meter.update(batch_time)
                global_batch_time_meter.update(batch_time)
                global_iter = (epoch - 1) * self.steps_per_epoch + processed_iter
                global_remain_iter = max(self.total_train_iters - global_iter, 0)
                remain_time = global_remain_iter * global_batch_time_meter.avg
                if (
                    processed_iter % self.log_freq == 0
                    or processed_iter == self.steps_per_epoch
                ):
                    info = (
                        f"Train: [{epoch}/{self.max_epoch}]"
                        f"[{processed_iter}/{self.steps_per_epoch}] "
                        f"Data {data_time_meter.val:.3f} ({data_time_meter.avg:.3f}) "
                        f"Batch {batch_time_meter.val:.3f} ({batch_time_meter.avg:.3f}) "
                        f"Remain {self._format_seconds(remain_time)} "
                    )
                    for key in loss_meters.keys():
                        info += f"{key}: {loss_meters[key].val:.4f} "
                    info += (
                        f"Accum {accum_counter if accum_counter > 0 else self.grad_accum_steps}/"
                        f"{self.grad_accum_steps} "
                    )
                    info += f"Lr: {self._current_lr():.6f}"
                    self.logger.info(info)
                iter_start = time.perf_counter()

        if loss_meters is None or loss_meters["loss"].count == 0:
            raise RuntimeError("No training batches were processed.")
        result = {key: meter.avg for key, meter in loss_meters.items()}
        result["optimizer_steps"] = optimizer_steps
        return result

    def validate(self) -> dict[str, float]:
        self.model.eval()
        metrics = ["val_mIoU", "val_mAcc", "val_allAcc"]
        if self.cfg.get("loss", {}).get("type", "SemanticBoundaryLoss") == "SemanticBoundaryLoss":
            metrics.extend(
                [
                    "val_loss_mask",
                    "val_loss_vec",
                    "val_loss_strength",
                    "mask_precision",
                    "mask_recall",
                    "mask_f1",
                    "vec_error_masked",
                    "strength_error_masked",
                ]
            )
        else:
            metrics.append("val_loss_semantic")
        metric_meters = {key: AverageMeter() for key in metrics}
        max_batches = self.trainer_cfg.get("max_val_batches")
        num_classes = int(self.cfg["model"]["num_classes"])
        semantic_intersection = torch.zeros(num_classes, dtype=torch.float64)
        semantic_union = torch.zeros(num_classes, dtype=torch.float64)
        semantic_target = torch.zeros(num_classes, dtype=torch.float64)

        self.logger.info(">>>>>>>>>>>>>>>> Start Validation >>>>>>>>>>>>>>>>")
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                batch = self._move_batch_to_device(batch)
                self._ensure_cpu_backbone(batch)
                output = self.model(self._forward_input_from_batch(batch))
                metric_dict = self.evaluator(**self._build_eval_inputs(output, batch))
                detached = self._detach_scalar_dict(metric_dict)
                semantic_intersection += metric_dict["semantic_intersection"].detach().cpu().double()
                semantic_union += metric_dict["semantic_union"].detach().cpu().double()
                semantic_target += metric_dict["semantic_target"].detach().cpu().double()
                for key in metrics:
                    metric_meters[key].update(detached[key])
                processed_iter = batch_idx + 1
                if processed_iter % self.val_log_freq == 0 or processed_iter == (
                    min(len(self.val_loader), max_batches) if max_batches is not None else len(self.val_loader)
                ):
                    if "val_loss_mask" in metric_meters:
                        self.logger.info(
                            "Val/Test: [{iter}/{max_iter}] val_loss_mask: {loss_mask:.4f} "
                            "val_loss_vec: {loss_vec:.4f} val_loss_strength: {loss_strength:.4f} "
                            "mIoU: {miou:.4f} mAcc: {macc:.4f} allAcc: {allacc:.4f}".format(
                                iter=processed_iter,
                                max_iter=min(len(self.val_loader), max_batches) if max_batches is not None else len(self.val_loader),
                                loss_mask=metric_meters["val_loss_mask"].val,
                                loss_vec=metric_meters["val_loss_vec"].val,
                                loss_strength=metric_meters["val_loss_strength"].val,
                                miou=metric_meters["val_mIoU"].val,
                                macc=metric_meters["val_mAcc"].val,
                                allacc=metric_meters["val_allAcc"].val,
                            )
                        )
                    else:
                        self.logger.info(
                            "Val/Test: [{iter}/{max_iter}] val_loss_semantic: {loss_semantic:.4f} "
                            "mIoU: {miou:.4f} mAcc: {macc:.4f} allAcc: {allacc:.4f}".format(
                                iter=processed_iter,
                                max_iter=min(len(self.val_loader), max_batches) if max_batches is not None else len(self.val_loader),
                                loss_semantic=metric_meters["val_loss_semantic"].val,
                                miou=metric_meters["val_mIoU"].val,
                                macc=metric_meters["val_mAcc"].val,
                                allacc=metric_meters["val_allAcc"].val,
                            )
                        )

        if metric_meters["val_mIoU"].count == 0:
            raise RuntimeError("No validation batches were processed.")
        summary = {key: meter.avg for key, meter in metric_meters.items()}
        per_class_iou, per_class_acc = self._compute_per_class_from_stats(
            semantic_intersection.float(),
            semantic_union.float(),
            semantic_target.float(),
        )
        summary["semantic_iou_per_class"] = per_class_iou
        summary["semantic_acc_per_class"] = per_class_acc
        self.logger.info(
            "Val result: mIoU/mAcc/allAcc {miou:.4f}/{macc:.4f}/{allacc:.4f}.".format(
                miou=summary["val_mIoU"],
                macc=summary["val_mAcc"],
                allacc=summary["val_allAcc"],
            )
        )
        for class_id in range(num_classes):
            class_name = (
                self.class_names[class_id]
                if self.class_names is not None and class_id < len(self.class_names)
                else f"Class_{class_id}"
            )
            self.logger.info(
                "{name} Result: iou/accuracy {iou:.4f}/{acc:.4f}".format(
                    name=class_name,
                    iou=float(per_class_iou[class_id]),
                    acc=float(per_class_acc[class_id]),
                )
            )
        self.logger.info("<<<<<<<<<<<<<<<<< End Validation <<<<<<<<<<<<<<<<<")
        return summary

    def save_checkpoint(self, filename: str, epoch: int, val_metrics: dict[str, float]):
        filename = os.path.join(self.model_dir, filename)
        checkpoint = dict(
            epoch=epoch,
            model_state_dict=self.model.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            scheduler_state_dict=(
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            best_val_miou=self.best_val_miou,
            val_metrics=val_metrics,
        )
        self.logger.info(f"Saving checkpoint to: {filename}")
        torch.save(checkpoint, filename + ".tmp")
        os.replace(filename + ".tmp", filename)
        return filename

    def run(self):
        self.logger.info(">>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>")
        self.logger.info(f"Smoke mode: {self.smoke_mode}")
        self.logger.info(
            f"Total epoch: {self.total_epoch}, Eval epoch: {self.max_epoch}, Train loop: {self.train_loop}"
        )
        self.logger.info(
            f"Base train steps per loop: {self.base_steps_per_epoch}, Train steps per displayed epoch: {self.steps_per_epoch}"
        )
        self.logger.info(f"Optimizer steps per epoch: {self.optimizer_steps_per_epoch}")
        self.logger.info(f"Total train iterations: {self.total_train_iters}")
        if self.start_epoch > self.max_epoch:
            self.logger.info(
                f"Start epoch {self.start_epoch} is already beyond max epoch {self.max_epoch}. Nothing to run."
            )
            return
        for epoch in range(self.start_epoch, self.max_epoch + 1):
            train_metrics = self.train_one_epoch(epoch)
            val_metrics = self.validate()

            current_miou = val_metrics["val_mIoU"]
            is_best = current_miou > self.best_val_miou
            if is_best:
                self.best_val_miou = current_miou
                self.logger.info(
                    "Best validation mIoU updated to: {:.4f}".format(
                        self.best_val_miou
                    )
                )
            last_path = self.save_checkpoint("model_last.pth", epoch, val_metrics)
            if is_best:
                best_path = self.save_checkpoint("model_best.pth", epoch, val_metrics)
            else:
                best_path = os.path.join(self.model_dir, "model_best.pth")

            if self.save_freq and epoch % int(self.save_freq) == 0:
                self.save_checkpoint(f"epoch_{epoch}.pth", epoch, val_metrics)

            if "loss_mask" in train_metrics:
                self.logger.info(
                    "Train result: loss={loss:.4f} loss_semantic={loss_semantic:.4f} "
                    "loss_mask={loss_mask:.4f} loss_vec={loss_vec:.4f} "
                    "loss_strength={loss_strength:.4f} optimizer_steps={optimizer_steps}".format(
                        **train_metrics
                    )
                )
            elif "loss_lovasz" in train_metrics:
                self.logger.info(
                    "Train result: loss={loss:.4f} loss_semantic={loss_semantic:.4f} "
                    "loss_lovasz={loss_lovasz:.4f} optimizer_steps={optimizer_steps}".format(
                        **train_metrics
                    )
                )
            else:
                self.logger.info(
                    "Train result: loss={loss:.4f} loss_semantic={loss_semantic:.4f} "
                    "optimizer_steps={optimizer_steps}".format(**train_metrics)
                )
            self.logger.info(f"Current val_mIoU: {current_miou:.4f}")
            self.logger.info(f"Current best_val_mIoU: {self.best_val_miou:.4f}")
            self.logger.info(f"Checkpoint last: {last_path}")
            self.logger.info(f"Checkpoint best: {best_path}")
