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

        image = torch.from_numpy(image).float().permute(2, 0, 1)
        mask = torch.from_numpy(mask).float()

        # Ensure mask is (1, H, W)
        if mask.ndim == 3:      # (H, W, 1)
            mask = mask.permute(2, 0, 1)
        elif mask.ndim == 2:    # (H, W)
            mask = mask.unsqueeze(0)

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