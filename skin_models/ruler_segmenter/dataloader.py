import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as tf
import random
import os
import torchvision.transforms.functional as F
from PIL import Image, ImageOps

WIDTH = HEIGHT = 512

class CustomImageDataset(Dataset):
    def __init__(self, img_folder, mask_folder, transforms):
        super().__init__()
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        self.img_names = [f for f in os.listdir(img_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
        self.transforms = transforms

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        img_name = self.img_names[index]
        img_path = os.path.join(self.img_folder, img_name)
        mask_path = os.path.join(self.mask_folder, img_name)

        try:
            image = ImageOps.exif_transpose(Image.open(img_path).convert('RGB'))
            mask = ImageOps.exif_transpose(Image.open(mask_path).convert('L'))
        except:
            # Handle exception (e.g., file not found)
            return self.__getitem__(np.random.choice(len(self.img_names)))

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

        return image, mask

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
    
class RunImageDataset(Dataset):
    def __init__(self, img_folder):
        super().__init__()
        self.img_folder = img_folder
        self.img_names = [f for f in os.listdir(img_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        img_name = self.img_names[index]
        img_location = os.path.join(self.img_folder, img_name)

        try:
            image = ImageOps.exif_transpose(Image.open(img_location).convert('RGB'))

        except:
            random_idx = np.random.choice(len(self.img_names))
            with open('none_importable_images.txt','a+') as fh:
                fh.write(img_location +', ' + self.img_names[random_idx] + '\n')                
            return self.__getitem__(random_idx)
        
        image_width, image_height = image.size

        image = tf.Resize((HEIGHT, WIDTH))(image)
        image = tf.ToTensor()(image)
        image = tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)

        return image, image_height, image_width, img_location