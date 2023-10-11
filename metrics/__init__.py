import torch

from .lpips import lpips
from .ssim import ssim


def psnr(img1, img2):
    mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    torch_result = 20 * torch.log10(1.0 / torch.sqrt(mse))

    np_result = torch_result.detach().cpu().numpy().mean()

    return np_result
