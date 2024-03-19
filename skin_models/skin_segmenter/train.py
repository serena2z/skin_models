import argparse
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import os
import torchvision.models.segmentation
from dataloader import CustomImageDataset
    
def train(dataloader, save_dir, device, epochs, val_dataloader=None):
    device = torch.device(device)
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
    model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))
    model = model.to(device) 
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adjust the learning rate as needed
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)
    best_vloss = 1_000_000.
    best_vdice = 0.
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for epoch in range(epochs):
        print('EPOCH {}:'.format(epoch))
        model.train(True)
        avg_loss, avg_dice = train_one_epoch(dataloader, model, optimizer, device, loss_fn, scheduler)
        print('Training Loss: {}, Training Dice Score: {}'.format(avg_loss, avg_dice))
        model.train(False)
        if val_dataloader:
            avg_vloss, avg_vdice = validate(model, val_dataloader, device, loss_fn, epoch)
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = save_dir + '/model_best_loss' + '.pt' #best model based on validation loss
                torch.save(model.state_dict(), model_path)
            if avg_vdice > best_vdice:
                best_vdice = avg_vdice
                model_path = save_dir + '/model_best_dice' + '.pt' #best model based on dice score
                torch.save(model.state_dict(), model_path)
        else:
            model_path = save_dir + '/model_' + str(epoch) + '.pt'
            torch.save(model.state_dict(), model_path)
            
def validate(val_dataloader, model, device, loss_fn):
    model.eval()
    running_vloss = 0.0
    running_vdice = 0.0
    vdice_nan_count = 0
    with torch.no_grad():
        for i, vdata in enumerate(tqdm(val_dataloader)):
            vimages, vmasks = vdata
            vmasks = vmasks.squeeze(1).long()
            vimages = vimages.to(device)
            vmasks = vmasks.to(device)
            voutputs = model(vimages)['out']
            vloss = loss_fn(voutputs, vmasks)
            running_vloss += vloss.item() * vimages.size(0)
            for i in range(voutputs.shape[0]):
                vseg = torch.argmax(voutputs[i], axis=0).cpu().detach().numpy()
                vgt = vmasks[i].detach().cpu().numpy()
                vdice = dice_score(vseg, vgt)
                if ~np.isnan(vdice):
                    running_vdice += vdice
                else:
                    vdice_nan_count = vdice_nan_count + 1
    avg_vloss = running_vloss / len(val_dataloader.dataset)
    avg_vdice = running_vdice / (len(val_dataloader.dataset) - vdice_nan_count)
    print('Validation Loss: {}, Validation Dice Score: {}'.format(avg_vloss, avg_vdice))
    return avg_vloss, avg_vdice

def train_one_epoch(dataloader, model, optimizer, device, loss_fn, scheduler):
    running_loss = 0.
    avg_loss = 0.
    total_dice = 0.
    avg_dice = 0.
    dice_nan_count = 0
    for i, data in enumerate(tqdm(dataloader)):
        images, masks = data
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
            dice = dice_score(seg, gt)
            if ~np.isnan(dice):
                total_dice += dice
            else:
                dice_nan_count = dice_nan_count + 1
    scheduler.step()
    avg_loss = running_loss / len(dataloader.dataset)
    avg_dice = total_dice / (len(dataloader.dataset) - dice_nan_count)
    return avg_loss, avg_dice

def dice_score(seg, gt, k=1):
    dicey = np.sum(seg[gt==k])*2.0 / (np.sum(seg) + np.sum(gt)) 
    return dicey

def main():
    parser = argparse.ArgumentParser(description="Semantic Segmentation Training with DeepLabV3")
    parser.add_argument("--img_folder", type=str, default="./images", help="Path to the image folder")
    parser.add_argument("--mask_folder", type=str, default="./masks", help="Path to the mask folder")
    parser.add_argument("-transforms", action="store_true", help="Apply data augmentation")
    parser.add_argument("--device", type=str, default="cpu", help="Device for training (cuda:0 or cpu)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for dataloader")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for dataloader")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--save_dir", type=str, default="./models", help="Path to save the trained model")

    # Add validation arguments
    parser.add_argument("--val", action="store_true", help="Use validation dataset")
    parser.add_argument("--val_img_folder", type=str, default="./val_images", help="Path to the validation image folder")
    parser.add_argument("--val_mask_folder", type=str, default="./val_masks", help="Path to the validation mask folder")
    
    args = parser.parse_args()

    img_folder = args.img_folder
    mask_folder = args.mask_folder
    transforms = args.transforms
    device = args.device
    batch_size = args.batch_size
    num_workers = args.num_workers
    epochs = args.epochs
    save_dir = args.save_dir
    val = args.val
    val_img_folder = args.val_img_folder
    val_mask_folder = args.val_mask_folder

    dataset = CustomImageDataset(img_folder, mask_folder, transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    if val:
        val_dataset = CustomImageDataset(val_img_folder, val_mask_folder, transforms)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        train(dataloader, save_dir, device, epochs, val_dataloader)
    else:
        train(dataloader, save_dir, device, epochs)

if __name__ == "__main__":
    main()
