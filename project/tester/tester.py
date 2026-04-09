"""Fragment-based tester for sbf-net semantic segmentation.

Follows the upstream Pointcept SemSegTester pattern:
  1. Load checkpoint into model
  2. Build test dataloader with test_mode=True (fragment + TTA)
  3. For each scene: accumulate softmax votes across fragments
  4. Compute per-class mIoU / mAcc / allAcc

Usage:
    python scripts/train/test.py --config <config> --weight <checkpoint>
"""

from __future__ import annotations

import os
import time
from collections import OrderedDict
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import project.datasets  # noqa: F401 – register BFDataset
import project.models  # noqa: F401 – register models
import project.transforms  # noqa: F401 – register transforms
from pointcept.datasets import build_dataset
from pointcept.datasets.utils import collate_fn
from pointcept.models import build_model
from pointcept.utils.config import ConfigDict
from project.utils import create_logger


class SemanticBoundaryTester:
    """Single-GPU fragment-based tester with test-time augmentation."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.work_dir = cfg["work_dir"]
        self.weight = cfg["weight"]
        self.num_classes = int(cfg["data"]["num_classes"])
        self.ignore_index = int(cfg["data"].get("ignore_index", -1))
        self.class_names = cfg["data"].get("names")

        os.makedirs(self.work_dir, exist_ok=True)
        self.logger = create_logger(
            "sbf_net.tester", os.path.join(self.work_dir, "test.log")
        )
        self.logger.info("=> Loading config ...")
        self.logger.info(f"Save path: {self.work_dir}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Weight: {self.weight}")

        self.logger.info("=> Building model ...")
        self.model = self._build_model()

        self.logger.info("=> Building test dataset & dataloader ...")
        self.test_loader = self._build_test_loader()

    def _build_model(self) -> torch.nn.Module:
        model = build_model(self.cfg["model"]).to(self.device)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Num params: {n_parameters}")

        if not os.path.isfile(self.weight):
            raise FileNotFoundError(f"No checkpoint found at '{self.weight}'")

        self.logger.info(f"Loading weight at: {self.weight}")
        checkpoint = torch.load(self.weight, map_location=self.device, weights_only=False)

        state_dict = self._extract_model_state_dict(checkpoint)
        model.load_state_dict(state_dict, strict=True)

        epoch = checkpoint.get("epoch", "?")
        self.logger.info(f"=> Loaded weight (epoch {epoch})")
        return model

    @staticmethod
    def _extract_model_state_dict(checkpoint: dict) -> dict:
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if "state_dict" in checkpoint:
            sd = checkpoint["state_dict"]
            return OrderedDict(
                (k[7:] if k.startswith("module.") else k, v) for k, v in sd.items()
            )
        raise KeyError("No model weights found in checkpoint.")

    def _build_test_loader(self) -> DataLoader:
        test_cfg = self.cfg["data"]["test"]
        # DefaultDataset expects test_cfg with attribute access (ConfigDict),
        # but our config is a plain dict from runpy. Convert it.
        if isinstance(test_cfg.get("test_cfg"), dict):
            test_cfg["test_cfg"] = ConfigDict(test_cfg["test_cfg"])
        test_dataset = build_dataset(test_cfg)
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.cfg.get("test_num_workers", 4),
            pin_memory=True,
            collate_fn=lambda batch: batch,
        )
        return test_loader

    def test(self):
        self.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.model.eval()

        save_path = os.path.join(self.work_dir, "result")
        os.makedirs(save_path, exist_ok=True)

        intersection_total = np.zeros(self.num_classes, dtype=np.float64)
        union_total = np.zeros(self.num_classes, dtype=np.float64)
        target_total = np.zeros(self.num_classes, dtype=np.float64)
        record = {}
        batch_time_sum = 0.0

        for idx, batch_list in enumerate(self.test_loader):
            start = time.time()
            data_dict = batch_list[0]  # batch_size=1
            fragment_list = data_dict.pop("fragment_list")
            segment = data_dict.pop("segment")
            data_name = data_dict.pop("name")

            pred_save_path = os.path.join(save_path, f"{data_name}_pred.npy")

            if os.path.isfile(pred_save_path):
                self.logger.info(
                    f"{idx + 1}/{len(self.test_loader)}: {data_name}, loaded cached pred."
                )
                pred = np.load(pred_save_path)
                if "origin_segment" in data_dict:
                    segment = data_dict["origin_segment"]
            else:
                pred = torch.zeros(
                    (segment.size, self.num_classes), device=self.device
                )

                for frag_idx in range(len(fragment_list)):
                    input_dict = collate_fn([fragment_list[frag_idx]])
                    for key in input_dict:
                        if isinstance(input_dict[key], torch.Tensor):
                            input_dict[key] = input_dict[key].to(
                                self.device, non_blocking=True
                            )

                    idx_part = input_dict["index"]
                    with torch.no_grad():
                        output = self.model(
                            {
                                "coord": input_dict["coord"],
                                "grid_coord": input_dict["grid_coord"],
                                "feat": input_dict["feat"],
                                "offset": input_dict["offset"],
                            }
                        )
                        pred_part = F.softmax(output["seg_logits"], dim=-1)

                    bs = 0
                    for be in input_dict["offset"]:
                        pred[idx_part[bs:be], :] += pred_part[bs:be]
                        bs = be

                    self.logger.info(
                        f"Test: {idx + 1}/{len(self.test_loader)}-{data_name}, "
                        f"Fragment: {frag_idx + 1}/{len(fragment_list)}"
                    )

                pred = pred.argmax(dim=1).cpu().numpy()

                # map back to original resolution if voxelized
                if "origin_segment" in data_dict:
                    assert "inverse" in data_dict
                    pred = pred[data_dict["inverse"]]
                    segment = data_dict["origin_segment"]

                np.save(pred_save_path, pred)

            # per-scene metrics
            intersection, union, target = self._intersection_and_union(
                pred, segment, self.num_classes, self.ignore_index
            )
            intersection_total += intersection
            union_total += union
            target_total += target
            record[data_name] = dict(
                intersection=intersection, union=union, target=target
            )

            mask = union != 0
            iou = np.mean(intersection[mask] / (union[mask] + 1e-10)) if mask.any() else 0.0
            acc = intersection.sum() / (target.sum() + 1e-10)
            m_iou = np.mean(
                intersection_total[union_total != 0]
                / (union_total[union_total != 0] + 1e-10)
            )
            m_acc = np.mean(
                intersection_total[target_total != 0]
                / (target_total[target_total != 0] + 1e-10)
            )

            elapsed = time.time() - start
            batch_time_sum += elapsed
            avg_time = batch_time_sum / (idx + 1)

            self.logger.info(
                f"Test: {data_name} [{idx + 1}/{len(self.test_loader)}]-{segment.size} "
                f"Batch {elapsed:.3f}s ({avg_time:.3f}s) "
                f"Accuracy {acc:.4f} ({m_acc:.4f}) "
                f"mIoU {iou:.4f} ({m_iou:.4f})"
            )

        # final summary
        iou_class = intersection_total / (union_total + 1e-10)
        accuracy_class = intersection_total / (target_total + 1e-10)
        mIoU = np.mean(iou_class[union_total != 0]) if (union_total != 0).any() else 0.0
        mAcc = np.mean(accuracy_class[target_total != 0]) if (target_total != 0).any() else 0.0
        allAcc = intersection_total.sum() / (target_total.sum() + 1e-10)

        self.logger.info(
            f"Test result: mIoU/mAcc/allAcc {mIoU:.4f}/{mAcc:.4f}/{allAcc:.4f}"
        )
        for i in range(self.num_classes):
            name = (
                self.class_names[i]
                if self.class_names and i < len(self.class_names)
                else f"Class_{i}"
            )
            self.logger.info(
                f"  {name}: iou/accuracy {iou_class[i]:.4f}/{accuracy_class[i]:.4f}"
            )
        self.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

        return dict(
            mIoU=mIoU,
            mAcc=mAcc,
            allAcc=allAcc,
            iou_class=iou_class,
            accuracy_class=accuracy_class,
            record=record,
        )

    @staticmethod
    def _intersection_and_union(
        pred: np.ndarray,
        target: np.ndarray,
        num_classes: int,
        ignore_index: int = -1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pred = pred.reshape(-1).astype(np.int64)
        target = target.reshape(-1).astype(np.int64)

        mask = target != ignore_index
        pred = pred[mask]
        target = target[mask]

        intersection = np.zeros(num_classes, dtype=np.float64)
        union = np.zeros(num_classes, dtype=np.float64)
        target_count = np.zeros(num_classes, dtype=np.float64)

        for cls in range(num_classes):
            pred_cls = pred == cls
            target_cls = target == cls
            intersection[cls] = (pred_cls & target_cls).sum()
            union[cls] = (pred_cls | target_cls).sum()
            target_count[cls] = target_cls.sum()

        return intersection, union, target_count
