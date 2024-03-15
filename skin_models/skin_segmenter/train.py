import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tf
import random
import torchvision.models.segmentation
import torchvision.transforms.functional as F
from PIL import Image, ImageOps

WIDTH = HEIGHT = 512

#Q1: Do we want to add error files for pixels that are not 0 or 1?
#Q2: Do we want to add validation split native to the training script? VS. Having them as separate arguments?
# Folder path works as well (if they upload a folder)

class CustomImageDataset(Dataset):
    def __init__(self, df, img_path, mask_path, transforms):
        super().__init__()
        self.df = df
        self.img_path = img_path
        self.mask_path = mask_path
        self.transforms = transforms

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        img_location = self.df[self.img_path].iloc[index]
        mask_location = self.df[self.mask_path].iloc[index]
        try:
            image = ImageOps.exif_transpose(Image.open(img_location).convert('RGB'))
            mask = ImageOps.exif_transpose(Image.open(mask_location).convert('L'))

        except:
            random_idx = np.random.choice(self.df.shape[0])
            with open('none_importable_images.txt','a+') as fh:
                fh.write(img_location +', ' + self.df[self.img_path].iloc[random_idx] + '\n')                
            return self.__getitem__(random_idx)
        image_width, image_height = image.size
        if self.transforms:
            image, mask = self.transform(image, mask)
        else: 
            image = tf.Resize((HEIGHT, WIDTH))(image)
            image = tf.ToTensor()(image)
            image = tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)

            mask = tf.Resize((HEIGHT, WIDTH), tf.InterpolationMode.NEAREST)(mask)
            mask_np = np.array(mask, dtype=np.uint8)
            mask_np = mask_np / np.max(np.abs(mask_np)) if ((len(np.unique(mask_np)) != 1) or (np.sum(mask_np) != 0)) else mask_np
            mask_np = mask_np.astype(np.float32)
            mask = tf.ToTensor()(mask_np)
        return image, mask, image_height, image_width, img_location
    
    def transform(self, image, mask):
        # RESIZE DISTORTION CORRECTION
        # randomly crop the longer dimension up to making the image square
        # prevents performance decrease due to distortion during resizing
        # and squishing of image along one dimension due it not being square
        if random.random() > 0.3:
            w,h = image.size
            l,t = (0,0)
            hw_diff = abs(w - h)
            offset = random.randint(0, hw_diff)
            if w<h:
                h =  h - offset
                t = int(offset/2)
            elif w>h:
                w = w - offset
                l = int(offset/2)
            image = F.crop(image,top = t, left = l, height = h, width = w)
            mask = F.crop(mask,top = t, left = l, height = h, width = w)
        # RESIZE
        image = tf.Resize((HEIGHT, WIDTH))(image)
        mask = tf.Resize((HEIGHT, WIDTH), tf.InterpolationMode.NEAREST)(mask)
        # RANDOM HORIZONTAL FLIP
        if random.random() > 0.5:
            image = F.hflip(image)
            mask = F.hflip(mask)
        # RANDOM VERTICAL FLIP
        if random.random() > 0.5:
            image = F.vflip(image)
            mask = F.vflip(mask)
        # RANDOM AFFINE - ROTATION, TRANSLATION, SCALE, SHEAR
        if random.random() > 0.3:
            degree, translate, scale, shear = tf.RandomAffine.get_params(degrees=[-90, 90], translate=[0.3, 0.3], scale_ranges=[0.75, 1.25], shears=[0, 0], img_size=(height, width))
            image = F.affine(image, degree, translate, scale, shear, interpolation=tf.InterpolationMode.BILINEAR)
            mask = F.affine(mask, degree, translate, scale, shear, interpolation=tf.InterpolationMode.NEAREST)
        # COLOR JITTER
        order, brightness, contrast, saturation, hue = tf.ColorJitter.get_params(brightness=[0.9, 1.1], contrast=[0.9, 1.1], saturation=[0.9, 1.1], hue=[-0.1, 0.1])
        # apply the color jitter in the order specified
        for i in order:
            if i == 0:
                image = F.adjust_brightness(image, brightness)
            elif i == 1:
                image = F.adjust_contrast(image, contrast)
            elif i == 2:
                image = F.adjust_saturation(image, saturation)
            elif i == 3:
                image = F.adjust_hue(image, hue)
        # TO TENSOR
        image = tf.ToTensor()(image)
        mask_np = np.array(mask,dtype=np.uint8)
        mask_np = mask_np/np.max(np.abs(mask_np)) if ((len(np.unique(mask_np)) != 1) | (np.sum(mask_np)!=0))  else mask_np
        mask_np = mask_np.astype(np.float32)
        mask_np = tf.ToTensor()(mask_np)
        # NORMALIZE
        image = tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)        
        return image, mask

def train(dataloader, save_dir, device, epochs, val_dataloader=None):
    device = torch.device(device)
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
    model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))
    model = model.to(device) 
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adjust the learning rate as needed
    best_vloss = 1_000_000.
    best_vdice = 0.
    for epoch in range(epochs):
        print('EPOCH {}:'.format(epoch))
        model.train(True)
        avg_loss, avg_dice = train_one_epoch(dataloader, model, optimizer, device, loss_fn)
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
    parser.add_argument("--data_path", type=str, default="./data.csv", help="Path to the dataset")
    parser.add_argument("--col_name", type=str, default="file_name", help="Name of the column containing the image file names")
    parser.add_argument("--mask_col_name", type=str, default="mask_file_name", help="Name of the column containing the mask file names")
    parser.add_argument("--save_dir", type=str, default="./masks", help="Directory to save segmentation masks")
    parser.add_argument("-transforms", action="store_true", help="Apply data augmentation")
    parser.add_argument("--device", type=str, default="cpu", help="Device for training (cuda:0 or cpu)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for dataloader")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for dataloader")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    # validation split
    parser.add_argument("-val", action="store_true", help="Add validation dataset")
    parser.add_argument("--val_data_path", type=str, default="./val_data.csv", help="Path to the validation dataset")
    parser.add_argument("--val_col_name", type=str, default="file_name", help="Name of the column containing the image file names")
    parser.add_argument("--val_mask_col_name", type=str, default="mask_file_name", help="Name of the column containing the mask file names")

    args = parser.parse_args()

    data_path = args.data_path
    col_name = args.col_name
    mask_col_name = args.mask_col_name
    save_dir = args.save_dir
    transforms = args.transforms
    device = args.device
    batch_size = args.batch_size
    num_workers = args.num_workers
    epochs = args.epochs
    val = args.val
    val_data_path = args.val_data_path
    val_col_name = args.val_col_name
    val_mask_col_name = args.val_mask_col_name

    dataset = CustomImageDataset(pd.read_csv(data_path), col_name, mask_col_name, transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    if val:
        val_dataset = CustomImageDataset(pd.read_csv(val_data_path), val_col_name, val_mask_col_name, transforms)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        train(dataloader, save_dir, device, epochs, val_dataloader)
    else:
        train(dataloader, save_dir, device, epochs)
            
if __name__ == "__main__":
    main()
