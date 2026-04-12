"""S3DIS dataset wrapper with boundary_mask + s_weight loading.

Same pattern as ZAHADataset / BFDataset: extends DefaultDataset to
additionally load boundary_mask_r060.npy and s_weight_r060_r120.npy.
"""

import os
from pathlib import Path

import numpy as np

from pointcept.datasets.builder import DATASETS
from pointcept.datasets.defaults import DefaultDataset


@DATASETS.register_module(force=True)
class S3DISBFDataset(DefaultDataset):

    def get_data(self, idx):
        data_dict = super().get_data(idx)
        data_path = Path(self.data_list[idx % len(self.data_list)])

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
