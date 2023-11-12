import os
from PIL import Image

from torch.utils.data import Dataset
import numpy as np
import random

class ImgtoMaskDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform= None, max_images=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.max_images = max_images
        self.images = os.listdir(image_dir)
        random.shuffle(self.images)
        self.images = self.images[:self.max_images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])

        mask_path = os.path.join(self.mask_dir, self.images[index])#.replace("c1", "c0") this for HeLA
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask > 0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask


class ImgToImgDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform= None, max_images=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.max_images = max_images
        self.images = os.listdir(image_dir)
        random.shuffle(self.images)
        self.images = self.images[:self.max_images]


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("RGB"))
        

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask