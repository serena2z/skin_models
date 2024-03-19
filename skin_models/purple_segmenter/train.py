import argparse
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torchvision.models.segmentation
import torchvision.transforms.functional as F
from dataloader import CustomImageDataset
from utils import get_image_patches, stitch_patches, drop_empty_patches, dice_score


def train_one_epoch(dataloader, model, optimizer, device, loss_fn, scheduler, add_empty=40):
    running_loss = 0.
    avg_loss = 0.
    avg_dice = 0.
    dice = []
    for i, data in enumerate(tqdm(dataloader)):
        images, masks = data
        images = get_image_patches(images)
        masks = get_image_patches(masks)
        images, masks = drop_empty_patches(images, masks, add_empty, shuffle=True,)
        masks = masks.squeeze(1).long()
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)['out']
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        for i in range(outputs.shape[0]): #iterating over batch
            seg = torch.argmax(outputs[i], axis=0).cpu().detach().numpy()
            gt = masks[i].detach().cpu().numpy()
            dice.append(dice_score(seg, gt))
    scheduler.step()
    avg_loss = running_loss / len(dice)
    avg_dice = np.sum(dice) / len(dice)
    return avg_loss, avg_dice


def validate(model, val_dataloader, loss_fn, device):
    model.eval()
    running_vloss = 0.0
    vdice = []
    vdice_f = [] #full image dice
    with torch.no_grad():
        for i, vdata in enumerate(tqdm(val_dataloader)):
            vimages, vmasks = vdata

            shape_f = vimages.shape
            vmasks_f = vmasks.clone().detach().squeeze(1).long()

            vimages = get_image_patches(vimages)
            vmasks = get_image_patches(vmasks)

            vmasks = vmasks.squeeze(1).long()

            vimages = vimages.to(device)
            vmasks = vmasks.to(device)
            voutputs = model(vimages)['out']
            vloss = loss_fn(voutputs, vmasks)
            running_vloss += vloss.item() * vimages.size(0)

            for i in range(voutputs.shape[0]):
                vseg = torch.argmax(voutputs[i], axis=0).cpu().detach().numpy()
                vgt = vmasks[i].detach().cpu().numpy()
                vdice.append(dice_score(vseg, vgt))

            voutputs_f = stitch_patches(voutputs, [shape_f[0], 2, shape_f[2], shape_f[3]])
            for i in range(voutputs_f.shape[0]):
                vseg_f = torch.argmax(voutputs_f[i], axis=0).cpu().detach().numpy()
                vgt_f = vmasks_f[i].detach().cpu().numpy()
                vdice_f.append(dice_score(vseg_f, vgt_f))
    avg_vloss = running_vloss / len(vdice)
    avg_vdice = sum(vdice) / len(vdice)
    avg_vdice_f = sum(vdice_f) / len(vdice_f)
    return avg_vloss, avg_vdice, avg_vdice_f


def train(dataloader, save_dir, device, epochs, add_empty, val_dataloader=None):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
    model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))
    model = model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-5)
    loss_fn = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

    best_vloss = 1_000_000.
    best_vdice = 0.
    best_vdice_f = 0.

    for epoch in range(epochs):
        print('EPOCH {}:'.format(epoch))
        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss, avg_dice = train_one_epoch(dataloader, model, optimizer, device, loss_fn, scheduler, add_empty)
        print('Training loss: {:.4f}, Training dice: {:.4f}'.format(avg_loss, avg_dice))
        # We don't need gradients on to do reporting
        model.train(False)

        # if save_dir does not exist, create it
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if val_dataloader:
            # Validate
            avg_vloss, avg_vdice, avg_vdice_f = validate(model, val_dataloader, loss_fn, device)
            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = os.path.join(save_dir, 'model_best_loss.pt')  # best model based on validation loss
                torch.save(model.state_dict(), model_path)

            if avg_vdice > best_vdice:
                best_vdice = avg_vdice
                model_path = os.path.join(save_dir, 'model_best_dice.pt')  # best model based on dice score
                torch.save(model.state_dict(), model_path)

            if avg_vdice_f > best_vdice_f:
                best_vdice_f = avg_vdice_f
                model_path = os.path.join(save_dir, 'model_best_dice_fullimage.pt')  # best model based on full image dice score
                torch.save(model.state_dict(), model_path)

        else:
            model_path = os.path.join(save_dir, 'model_{}.pt'.format(epoch))  # model based on epoch
            torch.save(model.state_dict(), model_path)


def main():
    parser = argparse.ArgumentParser(description="Train a segmentation model.")
    parser.add_argument("--img_folder", type=str, default="./images", help="Path to the image folder")
    parser.add_argument("--mask_folder", type=str, default="./masks", help="Path to the mask folder")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size for training.")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for data loading.")
    parser.add_argument("--save_dir", type=str, default="./model/", help="Directory to save the model.")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs.")
    parser.add_argument("--device", type=str, default="cpu", help="Device for training (cuda:0 or cpu)")
    parser.add_argument("--transforms", action="store_true", help="Apply data augmentation")
    parser.add_argument("--add_empty", type=int, default=40, help="Adds in empty masks (if there are any) for better generalization")
    parser.add_argument("--val", action="store_true", help="Add validation dataset")
    parser.add_argument("--val_img_folder", type=str, default="./val_images", help="Path to the validation image folder")
    parser.add_argument("--val_mask_folder", type=str, default="./val_masks", help="Path to the validation mask folder")

    args = parser.parse_args()

    img_folder = args.img_folder
    mask_folder = args.mask_folder
    save_dir = args.save_dir
    transforms = args.transforms
    device = args.device
    batch_size = args.batch_size
    num_workers = args.num_workers
    epochs = args.epochs
    add_empty = args.add_empty
    val = args.val
    val_img_folder = args.val_img_folder
    val_mask_folder = args.val_mask_folder

    dataset = CustomImageDataset(img_folder, mask_folder, transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    if val:
        val_dataset = CustomImageDataset(val_img_folder, val_mask_folder, transforms)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        train(dataloader, save_dir, device, epochs, add_empty, val_dataloader)
    else:
        train(dataloader, save_dir, device, epochs, add_empty)


if __name__ == "__main__":
    main()


