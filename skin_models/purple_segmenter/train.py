import argparse
import os
import torch
import torchvision.transforms as tf
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
import torchvision.models.segmentation
import torchvision.transforms.functional as F
from datetime import datetime

WIDTH=HEIGHT=224*5
patch_size = (224,224)

# Q: Add validation functionality?

def get_image_patches(image_tensor, patch_size=(150, 150)):
    b, c, h, w = image_tensor.size()
    patches = []
    for t in range(b):
        for i in range(0, h, patch_size[1]):
            for j in range(0, w, patch_size[0]):
                patch = image_tensor[t, :, i:i + patch_size[1], j:j + patch_size[0]]
                patches.append(patch)
    patches = torch.stack(patches, dim=0)
    return patches

def stitch_patches(patches, original_shape, patch_size=(150, 150)):
    b, c, h, w = original_shape
    reconstructed_images = torch.zeros((b, c, h, w))
    patch_idx = 0
    for img_idx in range(b):
        for i in range(0, h, patch_size[1]):
            for j in range(0, w, patch_size[0]):
                reconstructed_images[img_idx, :, i:i+patch_size[1], j:j+patch_size[0]] = patches[patch_idx]
                patch_idx += 1
    return reconstructed_images

def drop_empty_patches(images,masks,add_empty,empty_mask_value=0,shuffle=False,min_patches=2):
    #remove images patches with empty mask patches (note empty mask is if all values are 0)
    ''' drops empty patches, where all values in masks are equal to empty_mask_value,
        images = [b,c,h,w], images stack
        masks = [b,c,h,w], mask stack (note it must have a channel dimension)
        shuffle = True or False, shuffels remaining patches and masks
        min_patches = 2, Prevents returning empty tensor if all masks are empty
        add_empty = 0, adds in empty masks (if there are any) for better generalization
    '''

    idx_nonempty_masks = [x for x in range(len(masks)) if any(masks[x,:,:,:].unique()!=empty_mask_value)]
    
    add_npatches = 0

    # ensure there are at least min_patches # of patches
    if len(idx_nonempty_masks) < min_patches:
        add_npatches = min_patches - len(idx_nonempty_masks) 
    # adding empty patches for generalization
    if (add_empty > add_npatches):
        add_npatches += add_empty - add_npatches
    # ensuring that can't add more patches than there are
    if (len(idx_nonempty_masks) + add_npatches) > len(masks):
        add_npatches = len(masks) - len(idx_nonempty_masks)
    
    # adding empty patches
    idx_empty_masks = list(set(range(len(masks))) - set(idx_nonempty_masks))
    idx_addpatches = random.sample(idx_empty_masks,add_npatches)
    idx_nonempty_masks.extend(idx_addpatches)

    #shuffling
    if shuffle:
        random.shuffle(idx_nonempty_masks)
    else:
        idx_nonempty_masks.sort()

    return images[idx_nonempty_masks,:,:,:], masks[idx_nonempty_masks,:,:,:]

class CustomImageDataset(Dataset):
    def __init__(self, df, img_path='file_name', mask_path='mask_path', transforms=False):
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
            with open('none_importable_images.txt', 'a+') as fh:
                fh.write(img_location + ', ' + self.df[self.img_path].iloc[random_idx] + '\n')
            return self.__getitem__(random_idx)

        image_width, image_height = image.size

        if self.transforms is True:
            image, mask = self.transform(image, mask)
        else: 
            image = tf.Resize((HEIGHT, WIDTH))(image)
            image = tf.ToTensor()(image)
            image = tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)

            mask = tf.Resize((HEIGHT, WIDTH), tf.InterpolationMode.NEAREST)(mask)       
            mask_np = np.array(mask, dtype=np.uint8)
            mask_np = mask_np / np.max(np.abs(mask_np)) if len(np.unique(mask_np)) != 1 else mask_np
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
        image = tf.Resize((HEIGHT,WIDTH))(image)
        mask = tf.Resize((HEIGHT,WIDTH), tf.InterpolationMode.NEAREST)(mask)

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
        mask_np = mask_np/np.max(np.abs(mask_np)) if len(np.unique(mask_np)) != 1 else mask_np
        mask_np = mask_np.astype(np.float32)
        mask_np = tf.ToTensor()(mask_np)

        # NORMALIZE
        image = tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)        

        return image, mask_np

def dice_score(seg, gt, k=1):
    """Calculate the dice score for a segmentation and ground truth"""
    intersection = np.sum(seg[gt==k])*2.0
    denominator = (np.sum(seg) + np.sum(gt))
    if (intersection == 0) & (denominator == 0):
        return 1
    elif  (intersection != 0) & (denominator == 0):
        return 0
    else:
        return intersection / denominator
    
def train_one_epoch(dataloader, model, optimizer, device, loss_fn, scheduler, add_empty=40):
    running_loss = 0.
    avg_loss = 0.
    avg_dice = 0.
    dice = []
    for i, data in enumerate(tqdm(dataloader)):
        images, masks = data
        images = get_image_patches(images, patch_size)
        masks = get_image_patches(masks,patch_size)
        images, masks = drop_empty_patches(images,masks,add_empty,shuffle=True,)
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
            # calculate dice score
            seg = torch.argmax(outputs[i], axis=0).cpu().detach().numpy()
            gt = masks[i].detach().cpu().numpy()
            dice.append(dice_score(seg, gt))
    scheduler.step()
    # calculate loss for epoch
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

            vimages = get_image_patches(vimages, patch_size)
            vmasks = get_image_patches(vmasks,patch_size)

            # no need to drop empty patches during validation
            #vimages, vmasks = drop_empty_patches(vimages,vmasks,add_empty=4)
            
            #remove images patches with empty mask patches (note empty mask is if all values are 0)
            #idx_nonempty_vmasks = [x for x in range(len(vmasks)) if any(vmasks[x,:,:,:].unique()!=0)]
            #vimages = vimages[idx_nonempty_vmasks,:,:,:]
            #vmasks = vmasks[idx_nonempty_vmasks,:,:,:]

            vmasks = vmasks.squeeze(1).long()

            vimages = vimages.to(device)
            vmasks = vmasks.to(device)
            voutputs = model(vimages)['out']
            #voutputs = voutputs.squeeze(1)
            vloss = loss_fn(voutputs, vmasks)
            running_vloss += vloss.item() * vimages.size(0)

            # Dice score patches
            for i in range(voutputs.shape[0]):
                # calculate dice score
                vseg = torch.argmax(voutputs[i], axis=0).cpu().detach().numpy()
                vgt = vmasks[i].detach().cpu().numpy()
                vdice.append(dice_score(vseg, vgt))

            # Dice score full image
            voutputs_f = stitch_patches(voutputs,[shape_f[0],2,shape_f[2],shape_f[3]],patch_size)
            for i in range(voutputs_f.shape[0]):
                # calculate dice score
                vseg_f = torch.argmax(voutputs_f[i], axis=0).cpu().detach().numpy()
                vgt_f = vmasks_f[i].detach().cpu().numpy()
                vdice_f.append(dice_score(vseg_f,vgt_f))
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
        # We don't need gradients on to do reporting
        model.train(False)

        if val_dataloader:
        # Validate
            avg_vloss, avg_vdice, avg_vdice_f = validate(model, val_dataloader, loss_fn, device)
        # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = save_dir + '/model_best_loss' + '.pt' #best model based on validation loss
                torch.save(model.state_dict(), model_path)

            if avg_vdice > best_vdice:
                best_vdice = avg_vdice
                model_path = save_dir + '/model_best_dice' + '.pt' #best model based on dice score
                torch.save(model.state_dict(), model_path)

            if avg_vdice_f > best_vdice_f:
                best_vdice_f = avg_vdice_f
                model_path = save_dir + '/model_best_dice_fullimage' + '.pt' #best model based on dice score
                torch.save(model.state_dict(), model_path)

        else:
            model_path = save_dir + '/model_{}'.format(epoch) + '.pt'  #model based on epoch
            torch.save(model.state_dict(), model_path)


def main():
    parser = argparse.ArgumentParser(description="Train a segmentation model.")
    parser.add_argument("--data_path", type=str, default="./data.csv", help="Path to the dataset")
    parser.add_argument("--col_name", type=str, default="file_name", help="Name of the column containing the image file names")
    parser.add_argument("--mask_col_name", type=str, default="mask_file_name", help="Name of the column containing the mask file names")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size for training.")
    parser.add_argument("--num_workers", type=int, default=6, help="Number of workers for data loading.")
    parser.add_argument("--save_dir", type=str, default="./model/", help="Directory to save the model.")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs.")
    parser.add_argument("--device", type=str, default="cpu", help="Device for training (cuda:0 or cpu)")
    parser.add_argument("-transforms", action="store_true", help="Apply data augmentation")

    # add_empty = 0, adds in empty masks (if there are any) for better generalization
    parser.add_argument("--add_empty", type=int, default=40, help="Adds in empty masks (if there are any) for better generalization")

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
    add_empty = args.add_empty
    val = args.val
    val_data_path = args.val_data_path
    val_col_name = args.val_col_name
    val_mask_col_name = args.val_mask_col_name

    dataset = CustomImageDataset(pd.read_csv(data_path), col_name, mask_col_name, transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    if val:
        val_dataset = CustomImageDataset(pd.read_csv(val_data_path), val_col_name, val_mask_col_name, transforms)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        train(dataloader, save_dir, device, epochs, add_empty, val_dataloader)
    else:
        train(dataloader, save_dir, device, add_empty, epochs)
            
if __name__ == "__main__":
    main()
