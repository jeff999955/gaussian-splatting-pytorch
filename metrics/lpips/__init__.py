import torch

from .modules.lpips import LPIPS


def get_lpips_model(
    net_type: str = "vgg",
    version: str = "0.1",
) -> LPIPS:
    return LPIPS(net_type, version)


def lpips(
    x: torch.Tensor,
    y: torch.Tensor,
    evaluation_model: LPIPS,
):
    r"""Function that measures
    Learned Perceptual Image Patch Similarity (LPIPS).

    Arguments:
        x, y (torch.Tensor): the input tensors to compare.
        net_type (str): the network type to compare the features:
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.
    """
    return evaluation_model(x, y)
