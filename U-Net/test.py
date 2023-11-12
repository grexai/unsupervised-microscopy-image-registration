from distutils.command import config
from utils import (
    load_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
    save_predictions_as_imgs2,
)

import data_loading
from model import UNET
import torch.nn as nn
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from config import ConfigParams
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Compute the homography \
              between two images with the SuperPoint feature matches.')
    parser.add_argument('config_name', type=str)
    parser.add_argument('model_name', type=str)
    parser.add_argument('test_folder',type= str)
    args = parser.parse_args()
    cfg_name = args.config_name
    model_name = args.model_name
    configs = ConfigParams(cfg_name)
    arg_test_folder_dir= args.test_folder
    #  Hyperparameters

    LEARNING_RATE = configs.lr
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 1
    NUM_EPOCHS = configs.epochs
    NUM_WORKERS = 2
    IMAGE_HEIGHT = configs.image_size[0]
    IMAGE_WIDTH = configs.image_size[1]
    PIN_MEMORY = True
    LOAD_MODEL = configs.load_pretrained
    TRAIN_IMAGE_DIR = configs.train_image_dir
    TRAIN_MASK_DIR = configs.train_mask_dir

    VAL_IMG_DIR = configs.test_image_dir

    VAL_MASK_DIR = configs.test_mask_dir
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),

            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    _, val_loader = get_loaders(
        TRAIN_IMAGE_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    print(model_name)
    model = UNET(in_channels=3, out_channels=3,  features=configs.features).to(DEVICE)
    load_checkpoint(torch.load(f'{model_name}.pth.tar', map_location="cuda"), model)
    print(f'./test_{model_name.replace(".pth.tar","").replace("config","")}')

    print("something")
    if not os.path.exists(f'./test_{model_name.replace(".pth.tar","").replace("config","")}'):
        os.mkdir(f'./test_{model_name.replace(".pth.tar","").replace("config","")}')


    save_predictions_as_imgs(
                    val_loader, model, folder=f'./test_{model_name.replace(".pth.tar","").replace("config","")}', device=DEVICE
                )


