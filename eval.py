from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.testing_set import TestDataset
from metrics import lpips, psnr, ssim
from metrics.lpips import get_lpips_model
from metrics.ssim import create_window

evaluation_metrics = ["psnr", "ssim", "lpips"]


def evaluate(image_path: str, window, lpips_model, use_cuda: bool = True):
    try:
        dataset = TestDataset(image_path)
        dataloader = DataLoader(dataset, shuffle=False)

        result = defaultdict(list)

        # Set the window as the same type of the image
        img_tensor = dataset[0]["gt"]
        window = window.type_as(img_tensor)

        # move the window and model to the deivce
        if use_cuda:
            lpips_model = lpips_model.to("cuda")
            window = window.to("cuda")

        for batch in tqdm(dataloader):
            gt = batch["gt"]
            render = batch["render"]

            if use_cuda:
                gt = gt.to("cuda")
                render = render.to("cuda")

            # PSNR
            result["psnr"].append(psnr(gt, render))

            # SSIM
            result["ssim"].append(ssim(gt, render, 3, window=window).mean().item())

            # LPIPS
            result["lpips"].append(lpips(gt, render, lpips_model))

        for metric in evaluation_metrics:
            print(f"{metric.upper()}: {np.mean(result[metric])}")
    except Exception as e:
        print(f"Error: {e}")
        print(f"Failed to evaluate {image_path}")


def evaluate_all(image_paths: list[str], use_cuda: bool = True):
    # For SSIM
    window = create_window()

    # For LPIPS
    lpips_model = get_lpips_model()

    for image_path in image_paths:
        evaluate(
            image_path,
            window=window,
            lpips_model=lpips_model,
            use_cuda=use_cuda,
        )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument(
        "--image_paths", "-i", required=True, nargs="+", type=str, default=[]
    )
    parser.add_argument(
        "--cuda", "-c", action="store_true", help="Use CUDA if available"
    )
    args = parser.parse_args()

    use_cuda = args.cuda and torch.cuda.is_available()

    print(f"Using CUDA: {use_cuda}")
    evaluate_all(args.image_paths, use_cuda)
