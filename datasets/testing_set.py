import os

import torchvision
from PIL import Image
from torch.utils.data import Dataset


class TestDataset(Dataset):
    def __init__(self, directory, transform=None):
        """
        If the transform is specified, please add the ToTensor() transform to the end of the transform list.
        """
        self.directory = directory
        self.transform = transform or torchvision.transforms.ToTensor()

        # get the lists of 'gt' and 'renders' image files
        gt_images = sorted(os.listdir(os.path.join(self.directory, "gt")))
        render_images = sorted(os.listdir(os.path.join(self.directory, "renders")))

        assert gt_images == render_images, "'gt' and 'renders' images must be the same"

        # Since the images are the same, we can use the either one
        self.images = gt_images
        self.length = len(self.images)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # form the complete paths of 'gt' and 'renders' images
        image_name = self.images[idx]

        gt_path = os.path.join(self.directory, "gt", image_name)
        render_path = os.path.join(self.directory, "renders", image_name)

        # open and convert the images to RGB
        gt_image = Image.open(gt_path).convert("RGB")
        render_image = Image.open(render_path).convert("RGB")

        gt_image = self.transform(gt_image)
        render_image = self.transform(render_image)

        return {"gt": gt_image, "render": render_image}
