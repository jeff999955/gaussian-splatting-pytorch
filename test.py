import argparse

from datasets.ScanNet import ScanNetDataset
from metrics.ssim import create_window


def test_lpips():
    import torch

    from metrics import lpips
    from metrics.lpips import get_lpips_model

    n = 10
    x = torch.rand(n, 3, 256, 256)
    y = torch.rand(n, 3, 256, 256)

    model = get_lpips_model()
    res = lpips(x, y, model)
    print(res)


def test_ssim():
    import torch

    from metrics import ssim

    n = 10
    x = torch.rand(n, 3, 256, 256)
    y = torch.rand(n, 3, 256, 256)

    window = create_window()
    window = window.type_as(x)
    res = ssim(x, y, 3, window)
    print(res)


def test_psnr():
    import torch

    from metrics import psnr

    n = 10
    x = torch.rand(n, 3, 256, 256)
    y = torch.rand(n, 3, 256, 256)

    res = psnr(x, y)
    print(res)


def test_ScanNetDataset():
    args = argparse.Namespace(
        root_path="/scratch-ssd/scans/scene0241_01", split_setting="pointnerf"
    )
    train_dataset = ScanNetDataset(args)
    print("Path")
    print(train_dataset.root_path)
    print(train_dataset.image_path)
    print(train_dataset.pose_path)

    print("Size")
    print(train_dataset.original_size)
    print(train_dataset.target_size)

    print("Intrinsics")
    print(train_dataset.intrinsic)

    print("Cameras")
    print(len(train_dataset))
    print(train_dataset[0])

    print("Split")
    print(train_dataset.split)
    print(train_dataset.split_setting)

    print("Point cloud")
    print(train_dataset.point_cloud)


if __name__ == "__main__":
    test_ScanNetDataset()
