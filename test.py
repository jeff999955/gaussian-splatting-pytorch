def test_lpips():
    import torch

    from metrics import get_lpips_model, lpips

    n = 1
    x = torch.rand(n, 3, 256, 256)
    y = torch.rand(n, 3, 256, 256)

    model = get_lpips_model()
    res = lpips(x, y, model)
    print(res.mean().item())


if __name__ == "__main__":
    test_lpips()
