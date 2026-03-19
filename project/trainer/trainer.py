"""Minimal project-local trainer for stage-1 semantic boundary training."""

from __future__ import annotations

import os
from collections import defaultdict
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import project.datasets  # noqa: F401
import project.models  # noqa: F401
import project.transforms  # noqa: F401
from pointcept.datasets import build_dataset
from pointcept.datasets.utils import point_collate_fn
from pointcept.models import build_model
from project.evaluator import SemanticBoundaryEvaluator
from project.losses import SemanticBoundaryLoss


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
        self.optimizer_cfg = cfg["optimizer"]
        self.best_val_miou = float("-inf")
        self.smoke_mode = "actual_backbone_cuda"

        self.work_dir = self.trainer_cfg["work_dir"]
        os.makedirs(self.work_dir, exist_ok=True)

        self.train_dataset = build_dataset(cfg["data"]["train"])
        self.val_dataset = build_dataset(cfg["data"]["val"])
        self.train_loader = self._build_dataloader(self.train_dataset, training=True)
        self.val_loader = self._build_dataloader(self.val_dataset, training=False)

        self.model = build_model(cfg["model"]).to(self.device)
        self.loss_fn = SemanticBoundaryLoss().to(self.device)
        self.evaluator = SemanticBoundaryEvaluator()
        self.optimizer = self._build_optimizer()
        self._cpu_backbone_ready = False

    def _build_dataloader(self, dataset, training: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.trainer_cfg["batch_size"],
            shuffle=training,
            num_workers=self.trainer_cfg["num_workers"],
            collate_fn=partial(point_collate_fn, mix_prob=0),
            pin_memory=torch.cuda.is_available(),
            drop_last=training,
            persistent_workers=self.trainer_cfg["num_workers"] > 0,
        )

    def _build_optimizer(self):
        optimizer_type = self.optimizer_cfg["type"]
        kwargs = {
            key: value for key, value in self.optimizer_cfg.items() if key != "type"
        }
        if optimizer_type == "Adam":
            return torch.optim.Adam(self.model.parameters(), **kwargs)
        if optimizer_type == "SGD":
            return torch.optim.SGD(self.model.parameters(), **kwargs)
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

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
            if isinstance(value, torch.Tensor):
                result[key] = float(value.detach().cpu())
        return result

    @staticmethod
    def _average_logs(logs: list[dict[str, float]]) -> dict[str, float]:
        accum = defaultdict(float)
        for item in logs:
            for key, value in item.items():
                accum[key] += value
        return {key: value / max(len(logs), 1) for key, value in accum.items()}

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        train_logs = []
        max_batches = self.trainer_cfg.get("max_train_batches")

        for batch_idx, batch in enumerate(self.train_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = self._move_batch_to_device(batch)
            self._ensure_cpu_backbone(batch)

            self.optimizer.zero_grad(set_to_none=True)
            output = self.model(self._forward_input_from_batch(batch))
            loss_dict = self.loss_fn(
                seg_logits=output["seg_logits"],
                edge_pred=output["edge_pred"],
                segment=batch["segment"],
                edge=batch["edge"],
            )
            loss_dict["loss"].backward()
            self.optimizer.step()
            train_logs.append(self._detach_scalar_dict(loss_dict))

        if not train_logs:
            raise RuntimeError("No training batches were processed.")
        return self._average_logs(train_logs)

    def validate(self) -> dict[str, float]:
        self.model.eval()
        val_logs = []
        max_batches = self.trainer_cfg.get("max_val_batches")

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                batch = self._move_batch_to_device(batch)
                self._ensure_cpu_backbone(batch)
                output = self.model(self._forward_input_from_batch(batch))
                metric_dict = self.evaluator(
                    seg_logits=output["seg_logits"],
                    edge_pred=output["edge_pred"],
                    segment=batch["segment"],
                    edge=batch["edge"],
                )
                val_logs.append(self._detach_scalar_dict(metric_dict))

        if not val_logs:
            raise RuntimeError("No validation batches were processed.")
        return self._average_logs(val_logs)

    def save_checkpoint(self, filename: str, epoch: int, val_metrics: dict[str, float]):
        checkpoint = dict(
            epoch=epoch,
            model_state_dict=self.model.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            best_val_miou=self.best_val_miou,
            val_metrics=val_metrics,
        )
        torch.save(checkpoint, os.path.join(self.work_dir, filename))

    @staticmethod
    def _format_log(prefix: str, metrics: dict[str, float], keys: list[str]) -> str:
        values = [f"{key}={metrics[key]:.6f}" for key in keys]
        return f"{prefix}: " + ", ".join(values)

    def run(self):
        num_epochs = self.trainer_cfg["epochs"]
        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_one_epoch(epoch)
            val_metrics = self.validate()

            current_miou = val_metrics["val_mIoU"]
            self.save_checkpoint("model_last.pth", epoch, val_metrics)
            if current_miou > self.best_val_miou:
                self.best_val_miou = current_miou
                self.save_checkpoint("model_best.pth", epoch, val_metrics)

            print(f"epoch: {epoch}/{num_epochs}")
            print(f"device: {self.device}")
            print(f"smoke_mode: {self.smoke_mode}")
            print(
                self._format_log(
                    "train",
                    train_metrics,
                    [
                        "loss",
                        "loss_semantic",
                        "loss_mask",
                        "loss_vec",
                        "loss_strength",
                    ],
                )
            )
            print(
                self._format_log(
                    "val",
                    val_metrics,
                    [
                        "val_mIoU",
                        "val_mAcc",
                        "val_allAcc",
                        "val_loss_mask",
                        "val_loss_vec",
                        "val_loss_strength",
                        "mask_precision",
                        "mask_recall",
                        "mask_f1",
                        "vec_error_masked",
                        "strength_error_masked",
                    ],
                )
            )
            print(f"current_val_mIoU: {current_miou:.6f}")
            print(f"best_val_mIoU: {self.best_val_miou:.6f}")
            print(
                f"checkpoint_last: {os.path.join(self.work_dir, 'model_last.pth')}"
            )
            print(
                f"checkpoint_best: {os.path.join(self.work_dir, 'model_best.pth')}"
            )
