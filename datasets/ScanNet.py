import glob
import os
from copy import deepcopy

import numpy as np
from PIL import Image
from plyfile import PlyData

from .base import BaseDataset

# In ScanNet, it provides pose (camera to world) not extrinsic matrix(world to camera).


class ScanNetDataset(BaseDataset):
    def __init__(self, args, split="train"):
        self.root_path = args.root_path
        self.image_path = os.path.join(self.root_path, "color")
        self.pose_path = os.path.join(self.root_path, "pose")

        self.process_resize()
        self.target_size = (640, 480)  # w, h

        self.load_intrinsics(
            os.path.join(self.root_path, "intrinsic", "intrinsic_color.txt")
        )

        images = sorted(os.listdir(self.image_path))
        self.cameras = [
            {
                "id": int(image.split(".")[0]),
                "image_path": os.path.join(self.image_path, image),
                "pose_path": os.path.join(self.pose_path, image.split(".")[0] + ".txt"),
            }
            for image in images
        ]
        self.cameras.sort(key=lambda x: x["id"])
        self.read_cameras()

        self.split = split
        self.set_train_test_split()

        self.load_point_cloud()

    def __len__(self):
        return len(self.train_cameras)

    def __getitem__(self, idx):
        return self.cameras[idx]

    def read_cameras(self):
        for camera in self.cameras:
            c2w = np.loadtxt(camera["pose_path"])
            camera.update({"pose": c2w})

    def set_train_test_split(self):
        if self.split_setting == "pointnerf":
            step = 5
            if self.split == "train":
                self.cameras = self.cameras[::step]
        elif self.split_setting == "mipnerf":
            llffhold = 8  # TODO: extract this to args
            if self.split == "train":
                self.cameras = [
                    c for idx, c in enumerate(self.cameras) if idx % llffhold != 0
                ]
            else:
                self.cameras = [
                    c for idx, c in enumerate(self.cameras) if idx % llffhold == 0
                ]

    def process_resize(self):
        original_image = Image.open(os.path.join(self.image_path, "0.jpg"))

        self.original_size = (original_image.width, original_image.height)

    def load_intrinsics(self, path):
        self.intrinsic = np.loadtxt(path)

        # Scale the f-X (i.e. width)
        self.intrinsic[0, :] *= self.target_size[0] / self.original_size[0]

        # Scale the f-Y (i.e. height)
        self.intrinsic[1, :] *= self.target_size[1] / self.original_size[1]

    def load_point_cloud(self):
        if self.loaded_iter:
            self.model.load_ply(
                os.path.join(
                    self.model_path,
                    "point_cloud",
                    "iteration_" + str(self.loaded_iter),
                    "point_cloud.ply",
                )
            )
        # TODO: Random Sample 100k points within the bounding box
        # elif args.random_init_points:
        #     n_points = scene_info.point_cloud.points.shape[0]
        #     point_cloud = BasicPointCloud.random(n_points)
        #     self.model.create_from_pcd(point_cloud, self.cameras_extent)
        #     print(f"Randomly initializing point cloud with {n_points} points!")
        else:
            ply_path = os.path.join(self.root_path, "pcd.ply")
            plydata = PlyData.read(ply_path)
            vertices = plydata["vertex"]
            positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
            colors = (
                np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T
                / 255.0
            )
            try:
                normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
            except:
                normals = np.zeros_like(positions)

            self.point_cloud = BasicPointCloud(
                points=positions, colors=colors, normals=normals
            )

            # TODO: Determine whether to use camera extent or not when loading point cloud
            # self.model.create_from_pcd(point_cloud, self.cameras_extent)
