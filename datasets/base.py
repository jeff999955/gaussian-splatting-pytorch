from typing import Literal

import numpy as np
from torch.utils.data import Dataset

from dtypes import PointCloud


class BaseDataset(Dataset):
    root_path: str
    image_path: str
    pose_path: str

    # Size, (w, h)
    original_size: tuple[int, int]
    target_size: tuple[int, int]

    # Camera intrinsic
    intrinsic: np.ndarray

    # Cameras
    cameras: list[dict]

    # Dataset split
    split: Literal["train", "test"]
    split_setting: str

    # Point cloud
    point_cloud: PointCloud

    def __init__(self, args):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def read_cameras(self):
        raise NotImplementedError
