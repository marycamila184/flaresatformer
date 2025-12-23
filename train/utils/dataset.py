import random
import numpy as np
import torch
from torch.utils.data import Dataset
import utils.processing as processing

random.seed(42)

class ImageMaskDataset(Dataset):
    def __init__(
        self,
        n_channels,
        bands,
        image_list,
        mask_list,
        image_size,
        target_resize=None,
        augment=False,
    ):
        self.n_channels = n_channels
        self.bands = bands
        self.image_list = image_list
        self.mask_list = mask_list
        self.image_size = image_size
        self.target_resize = target_resize
        self.augment = augment

        assert len(self.image_list) == len(self.mask_list), \
            "Image list and mask list must have same length"

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        mask_path = self.mask_list[idx]

        image = self.process_image(img_path)
        mask = self.process_mask(mask_path)

        if self.augment:
            image, mask = self.apply_augmentation(image, mask)

        # Convert to torch tensors
        # Image: HWC → CHW
        image = torch.from_numpy(image).float().permute(2, 0, 1)

        # Mask: HWC or HW → 1HW
        if mask.ndim == 2:
            mask = mask[..., None]
        mask = torch.from_numpy(mask).float().permute(2, 0, 1)

        return image, mask

    def process_image(self, path):
        return processing.load_image(
            path,
            self.n_channels,
            bands=self.bands,
            target_size=self.target_resize
        )

    def process_mask(self, path):
        return processing.load_mask(
            path,
            target_size=self.target_resize
        )

    def apply_augmentation(self, image, mask):
        if random.random() > 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()

        if random.random() > 0.5:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()

        return image, mask
