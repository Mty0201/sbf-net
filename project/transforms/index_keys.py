"""Project-local transform helpers for point-wise key synchronization."""

from collections.abc import Sequence

from pointcept.datasets.transform import TRANSFORMS

DEFAULT_INDEX_VALID_KEYS = [
    "coord",
    "color",
    "normal",
    "superpoint",
    "strength",
    "segment",
    "instance",
]


@TRANSFORMS.register_module()
class InjectIndexValidKeys:
    """Extend data_dict['index_valid_keys'] without overriding existing entries."""

    def __init__(self, keys):
        assert isinstance(keys, Sequence)
        self.keys = list(keys)

    def __call__(self, data_dict):
        existing = list(data_dict.get("index_valid_keys", DEFAULT_INDEX_VALID_KEYS))
        for key in self.keys:
            if key not in existing:
                existing.append(key)
        data_dict["index_valid_keys"] = existing
        return data_dict
