from typing import NamedTuple

import numpy as np


class PointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array

    @classmethod
    def random(cls, n: int = 100_000):
        points = np.random.rand(n, 3)
        colors = np.random.rand(n, 3)
        points[:, 0] = -10 + 20 * points[:, 0]
        points[:, 1] = 10 * points[:, 1]
        points[:, 2] = -10 + 20 * points[:, 2]

        normals = np.zeros((n, 3))
        return PointCloud(points=points, colors=colors, normals=normals)
