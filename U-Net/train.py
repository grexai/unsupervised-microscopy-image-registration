import os.path
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from config import ConfigParams
from model import UNET
from piqa import SSIM
import argparse
import wandb
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_img_img_loaders,
    save_predictions_as_imgs,
)



'''
TRAIN_IMAGE_DIR = 'd:/datasets/spheroid/202010_dataset/preselected_dataset/train/images/'
TRAIN_MASK_DIR = 'd:/datasets/spheroid/202010_dataset/preselected_dataset/train/binary/'
VAL_IMG_DIR = 'd:/datasets/spheroid/202010_dataset/preselected_dataset/val/images/'
VAL_MASK_DIR = 'd:/datasets/spheroid/202010_dataset/preselected_dataset/val/binary/'


TRAIN_IMAGE_DIR = '/storage01/grexai/dev/imreg/pytorch-CycleGAN-and-pix2pix/datasets/hela63_realigned/A/train/'   #bf
TRAIN_MASK_DIR = '/storage01/grexai/dev/imreg/pytorch-CycleGAN-and-pix2pix/datasets/hela63_realigned/B/train/'  #fl
VAL_IMG_DIR = '/storage01/grexai/dev/imreg/pytorch-CycleGAN-and-pix2pix/datasets/hela63_realigned/A/train/'
VAL_MASK_DIR = '/storage01/grexai/dev/imreg/pytorch-CycleGAN-and-pix2pix/datasets/hela63_realigned/B/train/'
'''


class TrainParams:
    def __init__(self, loss=0, epoch=0):
        self._loss_val = loss
        self._epoch = epoch

    def get_loss(self):
        return self._loss_val

    def set_loss(self, new_loss):
        self._loss_val = new_loss

    def get_epoch(self):
        return self._epoch

    def set_epoch(self, n):
        self._epoch = n


tp = TrainParams()


def train_fn(loader, model, optimizer, loss_fn, scaler, scheduler):
    loop = tqdm(loader)
    running_loss = 0   
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)

        # targets = targets.float().unsqueeze(1).to(device=DEVICE)
        targets = torch.permute(targets, (0, 3, 1, 2))
        targets = targets.float().to(device=DEVICE)

        # forward

        #with torch.cuda.amp.autocast():
        predictions = model(data)
        loss = loss_fn(predictions, targets)
        # backward
        # lrs = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 60], gamma=0.1)
        current_lr = optimizer.param_groups[0]['lr']
        optimizer.zero_grad()
        loss.backward()
        # lrs.step()
        optimizer.step()
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        tp.set_loss(loss.item())
        # twdm update
        running_loss += loss.item()
        loop.set_postfix(loss=loss.item(), lr=current_lr, epoch=tp.get_epoch())
    
    train_loss = running_loss/len(loader)
    wandb.log({"avg  loss": train_loss})   
    scheduler.step()     
        # return loss.item()


class SSIMLoss(SSIM):
    def forward(self, x, y):
        return -(1. - super().forward(x, y))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Unet training for image image translation')
    parser.add_argument('config_name', type=str)
    parser.add_argument('run_name', type=str)
    parser.add_argument('--epochs', type=int, required=False)
    parser.add_argument('--max_images', type=int, required=False)

    args = parser.parse_args()
    cfg_name = args.config_name
    run_name_w = args.run_name
    arg_n_epochs= args.epochs
    arg_max_images= args.max_images
    configs = ConfigParams(cfg_name)

    wandb.init(project='unet image translation')
    wandb.config = {
        "learning_rate": configs.lr,
        "epochs": configs.epochs,
        "batch_size": configs.batch_size
    }
    wandb.run.name = run_name_w
    wandb.run.save()
    #  Hyperparameters
    

    LEARNING_RATE = configs.lr
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = configs.batch_size
    NUM_EPOCHS = configs.epochs
    NUM_WORKERS = 2
    IMAGE_HEIGHT = configs.image_size[0]
    IMAGE_WIDTH = configs.image_size[1]
    PIN_MEMORY = True
    LOAD_MODEL = configs.load_pretrained
    TRAIN_IMAGE_DIR = configs.train_image_dir
    TRAIN_MASK_DIR = configs.train_mask_dir
    VAL_IMG_DIR = configs.val_image_dir
    VAL_MASK_DIR = configs.val_mask_dir
    print(arg_n_epochs)
    if arg_n_epochs:
        NUM_EPOCHS = arg_n_epochs
    
    cuda_device = torch.cuda.get_device_name(0)
    print(cuda_device)
    print(f"start training unet for {NUM_EPOCHS}")
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

    model = UNET(in_channels=3, out_channels=3, features=configs.features).to(DEVICE)
    # pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"number  paramters {pytorch_total_params}")
    print(f"number of trainable paramter {pytorch_total_trainable_params}")

    # loss_fn = nn.BCEWithLogitsLoss()
    # loss_fn = nn.BCELoss()
    #loss_fn = nn.MSELoss()

    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.L1Loss()
    # loss_fn = SSIMLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    multistep_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 400, 600, 800], gamma=0.1)

    train_loader, val_loader = get_img_img_loaders(
        TRAIN_IMAGE_DIR,
        TRAIN_MASK_DIR,
        arg_max_images,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        if os.path.exists(configs.model_name):
            print(f"loading: {configs.model_name}")
            load_checkpoint(torch.load(configs.model_name), model)
        else:
            print(f"{configs.model_name} doesnt exist")
    else:
        print("creating a new model")
    # check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()
    best_loss_value = tp.get_loss()
    
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler,multistep_scheduler)
        tp.set_epoch(epoch)
        if epoch == 0:
            best_loss_value = tp.get_loss()
        # save model
        # if tp.get_loss() < best_loss_value:
        #     best_loss_value = tp.get_loss()
        #     checkpoint = {
        #         "state_dict": model.state_dict(),
        #         "optimizer": optimizer.state_dict(),
        #         "loss": loss_fn.state_dict(),
        #     }
        #     save_checkpoint(checkpoint, filename=configs.model_name)
        #     if not os.path.exists(f"./{run_name_w}_val_images/"):
        #         os.mkdir(f"{run_name_w}_val_images/")
        #     save_predictions_as_imgs(
        #         val_loader, model, folder=f"{run_name_w}_val_images/", device=DEVICE
        #     )
        if epoch % 200 == 0 and epoch > 0:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": loss_fn.state_dict(),
            }
            save_checkpoint(checkpoint, filename=f'{run_name_w}.pth.tar')
            if not os.path.exists(f"./{run_name_w}_val_images/"):
                os.mkdir(f"{run_name_w}_val_images/")
            save_predictions_as_imgs(
               
                val_loader, model, folder=f"{run_name_w}_val_images/", device=DEVICE
            )
           
        """
        print(tp.get_loss())
        if tp.get_loss() < best_loss_value:
            print(f"current best value {best_loss_value} current loss: {tp.get_loss()}",)
            best_loss_value = tp.get_loss()
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": loss_fn.state_dict(),
            }
            print("saving checkpoint")
            save_checkpoint(checkpoint)
        """
       
            

        # print some example
