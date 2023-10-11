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


if __name__ == "__main__":
    test_lpips()
    test_ssim()
    test_psnr()
