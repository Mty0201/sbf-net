"""BF dataset wrapper registered into Pointcept's dataset registry."""

import os
from pathlib import Path

import numpy as np

from pointcept.datasets.builder import DATASETS
from pointcept.datasets.defaults import DefaultDataset


@DATASETS.register_module()
class BFDataset(DefaultDataset):
    """Minimal BF dataset wrapper following the S3DISDataset pattern."""

    def get_data(self, idx):
        data_dict = super().get_data(idx)
        data_path = Path(self.data_list[idx % len(self.data_list)])
        edge_path = data_path / "edge.npy"
        if edge_path.is_file():
            data_dict["edge"] = np.load(edge_path).astype(np.float32)
        return data_dict

    def get_data_name(self, idx):
        remain, sample_name = os.path.split(self.data_list[idx % len(self.data_list)])
        remain, split_name = os.path.split(remain)
        return f"{split_name}-{sample_name}"
