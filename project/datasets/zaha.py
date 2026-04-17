"""ZAHA dataset wrapper registered into Pointcept's dataset registry.

Loads LoFG2-remapped segment (5 classes) + boundary_mask + s_weight.
No color feature — ZAHA PCD drops rgb unconditionally.
"""

import os
from pathlib import Path

import numpy as np

from pointcept.datasets.builder import DATASETS
from pointcept.datasets.defaults import DefaultDataset


@DATASETS.register_module(force=True)
class ZAHADataset(DefaultDataset):

    VALID_SEGMENT_FILES = ("segment_lofg2.npy", "segment.npy")

    def get_data_list(self):
        data_list = super().get_data_list()
        return [p for p in data_list if not os.path.basename(p).startswith(".")]

    def get_data(self, idx):
        data_path = Path(self.data_list[idx % len(self.data_list)])

        data_dict = {}
        data_dict["name"] = self.get_data_name(idx)
        data_dict["split"] = self.get_split_name(idx)
        data_dict["coord"] = np.load(data_path / "coord.npy").astype(np.float32)
        data_dict["normal"] = np.load(data_path / "normal.npy").astype(np.float32)

        for seg_file in self.VALID_SEGMENT_FILES:
            seg_path = data_path / seg_file
            if seg_path.is_file():
                data_dict["segment"] = np.load(seg_path).reshape(-1).astype(np.int64)
                break

        boundary_mask_path = data_path / "boundary_mask_r060.npy"
        if boundary_mask_path.is_file():
            data_dict["boundary_mask"] = (
                np.load(boundary_mask_path).reshape(-1).astype(np.float32)
            )
        s_weight_path = data_path / "s_weight_r060_r120.npy"
        if s_weight_path.is_file():
            data_dict["s_weight"] = (
                np.load(s_weight_path).reshape(-1).astype(np.float32)
            )
        return data_dict

    def get_data_name(self, idx):
        remain, sample_name = os.path.split(self.data_list[idx % len(self.data_list)])
        remain, split_name = os.path.split(remain)
        return f"{split_name}-{sample_name}"
