# %%

import pandas as pd 
import torch
import matplotlib.pyplot as plt
#from shape import shape_to_mask
from PIL import Image, ImageOps
import os
import glob as glob
import numpy as np
import cv2
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import datetime
from datetime import datetime
import random
import torchvision.transforms.functional as F
#from skimage import io, transform, util
import torch.nn as nn

torch.manual_seed(12)
random.seed(12)
np.random.seed(12)

#os.chdir('/home/chris/Projects/2022_SkinProject/Code/models/segmentation/purple_segmenter/')

# %%

# Sample function to get patches (as per your version)
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

# Sample function to stitch the patches back together
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

def drop_empty_patches(images,masks,empty_mask_value=0,shuffle=False,min_patches=2,add_empty=0):
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
    

#############################################################################################################
#                                       Custom Loss Function                                                #
#############################################################################################################

#class MultiClassFocalLoss(nn.Module):
#    def __init__(self, alpha=0.5, gamma=2.0):
#        super(MultiClassFocalLoss, self).__init__()
#        self.alpha = alpha
#        self.gamma = gamma
#
#    def forward(self, inputs, targets):
#        # Apply softmax to inputs
#        probs = torch.nn.functional.softmax(inputs, dim=1)
#        
#        # Gather the probabilities of the true labels for each pixel
#        target_probs = probs.gather(1, targets.unsqueeze(1))
#        target_probs = target_probs.squeeze(1)
#
#        # Compute focal loss
#        bce = -torch.log(target_probs)
#        focal_weight = (1 - target_probs) ** self.gamma
#        loss = self.alpha * focal_weight * bce
#
#        return loss.mean()

class MultiClassFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, num_classes=None):
        '''alpha: list of values between 0 and 1, corresponding to each class
         The sparse classes should have a higher value (closer to zero) '''
        super(MultiClassFocalLoss, self).__init__()
        self.gamma = gamma
        if alpha is None:
            # If alpha is not given, we'll assign equal weights to all classes
            self.alpha = [1.0 / num_classes] * num_classes
        else:
            # If alpha is provided, it should be a list of weights for each class
            assert len(alpha) == num_classes, "Alpha size should be same as num_classes"
            self.alpha = alpha

    def forward(self, inputs, targets):
        # Apply softmax to inputs
        probs = torch.nn.functional.softmax(inputs, dim=1)
        
        # Gather the probabilities of the true labels for each pixel
        target_probs = probs.gather(1, targets.unsqueeze(1))
        target_probs = target_probs.squeeze(1)
        
        # Get the corresponding alpha values for each true class label
        alpha_tensor = torch.tensor(self.alpha).to(inputs.device)
        alphas = alpha_tensor[targets.long()]

        # Compute focal loss
        bce = -torch.log(target_probs)
        focal_weight = (1 - target_probs) ** self.gamma
        loss = alphas * focal_weight * bce

        return loss.mean()

#############################################################################################################
#                                       Custom DataLoader                                                   #
#############################################################################################################


class CustomImageDataset(Dataset):
    def __init__(self, df, img_path = 'file_name', mask_path = 'mask_path', transforms = True):
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
            # originally had this but it didn't work
            image = ImageOps.exif_transpose(Image.open(img_location).convert('RGB'))
            mask = ImageOps.exif_transpose(Image.open(mask_location).convert('L'))

            # check the unique values in the mask
            unique_values = np.unique(np.array(mask))
            if len(unique_values) > 2:
                # write to a file
                with open('mask_unique_values_skin.txt','a+') as fh:
                    fh.write(img_location + ',' + str(np.unique(np.array(mask),return_counts=True)) + '\n')

            # NEW CHANGES
            # image = cv2.imread(img_location)[:,:,0:3]
            # #print('image shape: ', image.shape)
            # mask = cv2.imread(mask_location)[:,:,0:3]
            # #print('mask shape: ',mask.shape)
            # AnnMap = np.zeros(image.shape[0:2],np.float32)
            # if mask is not None:  AnnMap[mask[:,:,0] == 1 ] = 1
            #print('AnnMap shape: ',AnnMap.shape)

        except:
            # choose another random image to load instead
            random_idx = np.random.choice(self.df.shape[0])
            with open('none_importable_images_skin.txt','a+') as fh:
                fh.write(img_location +',' + self.df[self.img_path].iloc[random_idx] + '\n')
            return self.__getitem__(random_idx)

        # if self.transforms is not None:
        #     image = self.transforms['image'](image)
        #     mask_np = self.transforms['mask'](mask_np)
        #     #AnnMap = self.transforms['mask'](AnnMap)

        if self.transforms is True:
            image, mask = self.transform(image, mask)
        else: 
            image = tf.Resize((height,width))(image)
            image = tf.ToTensor()(image)
            image = tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image) 
            mask = tf.Resize((height,width), tf.InterpolationMode.NEAREST)(mask)       
            mask_np = np.array(mask,dtype=np.uint8)
            mask_np = mask_np/np.max(np.abs(mask_np)) if len(np.unique(mask_np)) != 1 else mask_np
            mask_np = mask_np.astype(np.float32)
            mask = tf.ToTensor()(mask_np)

        # check the unique values in the mask
        unique_values = torch.unique(mask)
        if len(unique_values) > 2:
            # write to a file
            with open('torchtf_unique_values_skin.txt','a+') as fh:
                fh.write(img_location +',' + ',' + str(unique_values) + '\n')

        return image, mask

    def classes(self):
        return self.df[self.label_col].unique().tolist()

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
        image = tf.Resize((height,width))(image)
        mask = tf.Resize((height,width), tf.InterpolationMode.NEAREST)(mask)

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
# %%

#########################################################################################
#                      Single Epoch Training                                            #
#########################################################################################

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

def train_one_epoch(epoch_index):
    running_loss = 0.
    avg_loss = 0.
    avg_dice = 0.
    dice = []
    for i, data in enumerate(tqdm(dataloaders['train'])):
        # Every data instance is an input + label pair
        images, masks = data

        images = get_image_patches(images, patch_size)
        masks = get_image_patches(masks,patch_size)

        images, masks = drop_empty_patches(images,masks,shuffle=True,add_empty=40)

        #remove images patches with empty mask patches (note empty mask is if all values are 0)
        #idx_nonempty_masks = [x for x in range(len(masks)) if any(masks[x,:,:,:].unique()!=0)]
        #images = images[idx_nonempty_masks,:,:,:]
        #masks = masks[idx_nonempty_masks,:,:,:]
       
        # remove the channel dimension for mask
        masks = masks.squeeze(1).long()

        images = images.to(device)
        masks = masks.to(device)

        #print(images.shape, masks.shape)

        # Zero your gradients for every batch!
        optimizer.zero_grad()
        # Make predictions for this batch
        outputs = model(images)['out']
        # remove the extra dimension
        #outputs = outputs.squeeze(1)
        # Compute the loss and its gradients
        loss = loss_fn(outputs, masks)
        loss.backward()
        # Adjust learning weights
        optimizer.step()
        running_loss += loss.item() * images.size(0)

        for i in range(outputs.shape[0]): #iterating over batch
            # calculate dice score
            seg = torch.argmax(outputs[i], axis=0).cpu().detach().numpy()
            gt = masks[i].detach().cpu().numpy()
            dice.append(dice_score(seg, gt))

      # if i % 100 == 99:
      #    last_loss = running_loss / 100 # loss per batch
      #    print('  batch {} loss: {}'.format(i + 1, last_loss))
      #    running_loss = 0.
    scheduler.step()

    # calculate loss for epoch
    avg_loss = running_loss / len(dice)
    avg_dice = np.sum(dice) / len(dice)

    return avg_loss, avg_dice

# %%

#############################################################################################################
#                                                   MAIN                                                    #
#############################################################################################################

# -------------- MODEL AND DATALOADER PARAMETERS --------------
Learning_Rate=1e-5
width=height=224*5 #224*6 # changing image width and height
batchSize=5
numWorkers=6
patch_size = (224,224)
EPOCHS = 150


#width=height=224 # changing image width and height
#batchSize=2
#numWorkers=1

model_save_dir = './model/' # <<<

if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

# ------------ Make Dataframe ----------------------------
f = '/home/chris/Projects/2022_SkinProject/data/segmentation/purple_segmentation/20240102_purple_segmentations_combined1.csv' # <<<
df = pd.read_csv(f,index_col=0)
df['mask_path'] = df['purple_mask_path'] # <<<
df['file_name'] = df['image_path']

#df = df.iloc[0:20,:]

df_train = df[df.subset == 'train'].copy()
df_val = df[df.subset == 'val'].copy()

image_datasets = {'train':  CustomImageDataset(df_train),
                  'val':    CustomImageDataset(df_val,transforms=None)}


dataloaders = {'train': DataLoader(image_datasets['train'], batch_size=batchSize,shuffle=True, num_workers=numWorkers),
               'val': DataLoader(image_datasets['val'], batch_size=batchSize,shuffle=False, num_workers=numWorkers)}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
#class_names = image_datasets['train'].classes()


# %%
#-------------- Make model and set net and optimizer-------------------------------------
#print(torch.cuda.memory_summary(device=None, abbreviated=False))
#cuda_device = 0
#torch.cuda.set_device(cuda_device)
num_classes = 2
device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True) # Load net
model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1)) # Change final layer to 2 classes
model=model.to(device)
optimizer=torch.optim.Adam(params=model.parameters(),lr=Learning_Rate) # Create adam optimizer
loss_fn = torch.nn.CrossEntropyLoss() # Create loss function
#loss_fn = MultiClassFocalLoss(alpha=[0.5,0.5],num_classes=num_classes)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)
# one test w/ learning rate decay and one without


#------------------- Training loop ---------------------------------------------
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
epoch_number = 0

best_vloss = 1_000_000.
best_vdice = 0.
best_vdice_f = 0.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number))
    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss, avg_dice = train_one_epoch(epoch_number)
    # We don't need gradients on to do reporting
    model.train(False)

    running_vloss = 0.0
    vdice = []
    vdice_f = [] #full image dice
    with torch.no_grad():
        for i, vdata in enumerate(tqdm(dataloaders['val'])):
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
    print('LOSS train {} {} valid {} {} {}'.format(avg_loss, avg_dice, avg_vloss, avg_vdice, avg_vdice_f))
    # another metric to track for model selection - dice scores
    

    #print('LOSS train {} '.format(avg_loss))

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = model_save_dir + '/model_best_loss' + '.pt' #best model based on validation loss
        torch.save(model.state_dict(), model_path)

    if avg_vdice > best_vdice:
        best_vdice = avg_vdice
        model_path = model_save_dir + '/model_best_dice' + '.pt' #best model based on dice score
        torch.save(model.state_dict(), model_path)

    if avg_vdice_f > best_vdice_f:
        best_vdice_f = avg_vdice_f
        model_path = model_save_dir + '/model_best_dice_fullimage' + '.pt' #best model based on dice score
        torch.save(model.state_dict(), model_path)

    if (epoch_number+1) % 10 == 0:
        model_path = model_save_dir + '/model_{}'.format(epoch_number) + '.pt'  #model based on epoch
        torch.save(model.state_dict(), model_path)

    epoch_number += 1


# %%
