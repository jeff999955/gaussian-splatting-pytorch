from torch.utils.data import Dataset


class BaseDataset(Dataset):
    train_cameras: list
    test_cameras: list

    def __init__(self, args):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def read_cameras(self):
        raise NotImplementedError

    def getNerfppNorm(self):
        raise NotImplementedError
