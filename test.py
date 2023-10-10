from metrics.ssim import create_window


def test_lpips():
    import torch

    from metrics import lpips
    from metrics.lpips import get_lpips_model

    n = 1
    x = torch.rand(n, 3, 256, 256)
    y = torch.rand(n, 3, 256, 256)

    model = get_lpips_model()
    res = lpips(x, y, model)
    print(res.mean().item())


def test_ssim():
    import torch

    from metrics import ssim

    n = 1
    x = torch.rand(n, 3, 256, 256)
    y = torch.rand(n, 3, 256, 256)

    window = create_window()
    window = window.type_as(x)
    res = ssim(x, y, 3, window)
    print(res)


if __name__ == "__main__":
    test_lpips()
    test_ssim()
