from pickletools import uint8
import torch
import torchvision
import wandb

from data_loading import ImgToImgDataset
from data_loading import ImgtoMaskDataset
from torch.utils.data import DataLoader
from skimage import io
import cv2
import numpy as np

from tqdm import tqdm


def save_checkpoint(state, filename="more_features.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")

    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(
        train_dir,
        train_maskdir,
        val_dir,
        val_maskdir,
        batch_size,
        train_transform,
        val_transform,
        num_workers=4,
        pin_memory=True,
):
    train_ds = ImgtoMaskDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = ImgtoMaskDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def get_img_img_loaders(
        train_dir,
        train_maskdir,
        max_images,
        val_dir,
        val_maskdir,
        batch_size,
        train_transform,
        val_transform,
        num_workers=4,
        pin_memory=True,
):
    train_ds = ImgToImgDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
        max_images= max_images
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = ImgToImgDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            #  y = y.to(device) # float type
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                    (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct / num_pixels * 100:.2f}"
    )
    print(f"Dice score: {dice_score / len(loader)}")
    wandb.log({"acc": num_correct / num_pixels * 100})
    model.train()


def save_predictions_as_imgs2(
        loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    masks_arr = [] 
    img_arr = []
    pred_arr = []
    table = wandb.Table(columns=['ID', 'Image'])
    for idx, (x, y) in enumerate(tqdm(loader)):

        x = x.to(device=device)
        with torch.no_grad():
            # preds = model(x)
            preds = torch.sigmoid(model(x))

            preds = preds.float()

        torchvision.utils.save_image(
            preds, f"{folder}/prob_maps_{idx}.png"
        )
        torchvision.utils.save_image(
            x, f"{folder}/original_images{idx}.png"
        )

        # segmentations = torchvision.utils.draw_segmentation_masks(x, masks = masks.to(int))

        # print(x.dtype, y.dtype)
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")
        preds = (preds > 0.5)
        y = y.to(device).unsqueeze(1)
        for batch in range(preds.shape[0]):
            abatch = preds[batch, :, :, :]
            abatch = abatch.cpu()
            animage = x[batch, :, :, :]
            animage = animage.cpu()
            abatch = abatch.permute(1, 2, 0).detach().numpy()
            animageN = animage.permute(1, 2, 0).detach().numpy()
            # imgL2 = cv2.normalize(abatch, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            imgL1 = cv2.normalize(animageN, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            gtmask = y[batch, :, :, :]
            gtmask = gtmask.cpu()
            # print(y.shape)
            # print(abatch.shape)
            gtmask = gtmask.permute(1, 2, 0).detach().numpy()
            gtmask = gtmask.astype(np.uint8)
 
            imgL2 = abatch.astype(np.uint8)
            imgL1 = imgL1.astype(np.uint8)
            img_arr.append(imgL1)
            pred_arr.append(abatch.squeeze())
            masks_arr.append(gtmask.squeeze())
            # wandb.log(
            #    {"my_image_key" : wandb.Image(imgL1, masks={
            #    "predictions" : {
            #    "mask_data" : abatch.squeeze(),
            #    "class_labels" : {1: "mitotic"}
            #    },
            #    "ground_truth" : {
            #    "mask_data" : gtmask.squeeze(),
            #    "class_labels" : {1: "mitotic"}
            #    }
            #    })})

                  
            
            io.imsave(f"{folder}/{loader.dataset.images[idx]}", imgL2)
            ret, thresh = cv2.threshold(imgL2, 0, 255, 0)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(imgL1, contours, -1, (0, 255, 0), 1)
            # wandb.log({"img": [wandb.Image(imgL1, caption = "images")]})
            io.imsave(f"{folder}/{loader.dataset.images[idx]}", imgL1)
            
            # print(animage.dtype, abatch.dtype)
            # print(animageN)
            # animage = Image.fromarray(np.uint8(animageN)).convert('RGB')

            # print(animage.shape)
            # abatch = Image.fromarray(abatch.astype('uint8'), 'L')

            # abatch = Image.fromarray((abatch[0] * 255).astype(np.uint8)).convert('L')
            # Image.fromarray(animageN.astype('uint8')).save('result.png')
            # Image.fromarray(animageN.astype('uint8')).save('result.png')

            # thisContour = abatch.point(lambda p:p==1 and 255)
            # thisEdges   = thisContour.filter(ImageFilter.FIND_EDGES)
            # animageN[np.nonzero(np.array(thisEdges))] = (255,255,255)
            # print(animageN.shape,animageN.dtype)
            # Image.fromarray(animageN.astype('uint8')).save('result.png')
            # abatch = 
            # test1 = abatch.permute(1,2,0).detach().numpy()
            # animage = abatch.detach()
            # test1 = abatch.detach()
            # F.to_pil_image(animage)
            # animage = cv2.normalize(animage, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            # abatch = cv2.normalize(abatch, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        counter = 0
        for img, label,gt in zip(img_arr,pred_arr,masks_arr):
            mask_img = wandb.Image(img, masks = {
            "prediction" : {
            "mask_data" : label,
            "class_labels" : {1: "mitotic"}
            },
            "ground_truth" : {
            "mask_data" : gt,
            "class_labels" : {1: "mitotic"}
            },
            })
            table.add_data(counter, mask_img)
            counter = counter + 1
        counter = 0
        wandb.log({"Table" : table})
        masks_arr = [] 
        img_arr = []
        pred_arr = []
        
        # print(animage.dtype, abatch.dtype)
        # segmentations = torchvision.utils.draw_segmentation_masks(animage.uint8(), abatch.uint8(), alpha=0.8, colors="blue")
        # torchvision.utils.save_image(
        #    segmentations, f"{folder}/conf_50{idx}.png"
        # )
        # segmentations = torchvision.utils.draw_segmentation_masks(x, masks = masks.to(int))

        # print(y.shape)
        #   imgL2 = cv2.normalize(test1, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        #   imgL2 = imgL2.astype(np.uint8)
        # io.imsave(f"{folder}/{loader.dataset.images[idx]}", imgL2)
        # y = y.detach().numpy()
        # torchvision.utils.save_image(y, f"{folder}{idx}.png")
    
    model.train()

import time
def save_predictions_as_imgs(
        loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(tqdm(loader)):

        x = x.to(device=device)
        s = time.time()
        with torch.no_grad():
            # preds = model(x)
            preds = model(x)
            # preds = (preds > 0.5).float()
            preds = preds.float()
        e = time.time()    
        print(e-s)
        print(preds.shape[0])
        # torchvision.utils.save_image(
        # preds, f"{folder}/pred_{idx}.png"
        # )
        # torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")
        # imgL2 = L2[j].permute(1, 2, 0).detach().numpy()
        # print(preds.shape)
        
        for batch in range(preds.shape[0]):
            
            abatch = preds[batch, :, :, :]
            animage = x[batch, :,:,:]
            # print(abatch.shape)
            s = time.time()
            abatch = abatch.cpu()
            animage = animage.cpu()
            test1 = abatch.permute(1, 2, 0).detach().numpy()
            test2 = animage.permute(1, 2, 0).detach().numpy()
            # print(y.shape)
            e = time.time()
            imgL2 = cv2.normalize(test1, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            imgL2 = imgL2.astype(np.uint8)
            imgL1 = cv2.normalize(test2, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            imgL1 = imgL1.astype(np.uint8)
            #wandb.log({"img": [wandb.Image(imgL1, caption = "real")]})
            #wandb.log({"img": [wandb.Image(imgL2, caption = "fake")]})
            io.imsave(f"{folder}/{loader.dataset.images[idx]}", imgL2)
        # y = y.detach().numpy()
        # torchvision.utils.save_image(y, f"{folder}{idx}.png")

    model.train()
