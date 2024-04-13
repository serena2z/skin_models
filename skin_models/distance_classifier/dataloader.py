from PIL import Image, ImageOps
from torch.utils.data import Dataset
import random
import torchvision.transforms as tf
import torchvision.transforms.functional as F
import numpy as np
import random
import torch

np.random.seed(12)
random.seed(12)
torch.manual_seed(12)

class CustomImageDataset(Dataset):

    def __init__(self, df, training_stage = None, image_path = "image_path", gan_path = "gan_path", label_col = "pxcm"):
        super().__init__()
        self.df = df
        self.training_stage = training_stage
        self.image_path = image_path
        self.gan_path = gan_path
        self.label_col = label_col

    def __len__(self):
        return self.df.shape[0]
    
    def random_zoom_factor(self, zoom_min, zoom_max):
        return random.uniform(zoom_min, zoom_max) #what if we base random zoomin on distance instead?


    def zoom_in_limit(self, wh_min, pix_limit):
        '''wh_min = the length of the smaller side of the original image
        pix_limit = the lower limit '''
        return float(wh_min)/float(pix_limit)
    
    def zoom(self, original, x, y, zoom_factor,pixel_zoom=None,return_zoom_factor=False):
        'zoom_factor == zoom_in == 1/zoom_out'

        if pixel_zoom is None:
            side_length = (1.0/zoom_factor) * min(original.size)
        else:
            side_length = pixel_zoom

        left = x - side_length / 2.0
        right = x + side_length / 2.0
        top = y - side_length / 2.0
        bottom = y + side_length / 2.0

        cropped = original.crop((left, top, right, bottom))

        #actual zoom factor, due to rounding deviation
        actual_zoom_factor = float(np.min(original.size)) / float(cropped.size[np.argmin(original.size)])
        
        if not return_zoom_factor:
            return cropped
        else:
            return cropped, actual_zoom_factor
        
    def train_transform_zoom(self, image, label, x, y):# becuase this is the only one zooming
        
        #randomly rotate about lesion point
        if random.random() > 0.5:
            image = tf.RandomRotation(180,center=(x,y))(image)

        #zoom
        zoom_min = 0.5 
        zoom_max = self.zoom_in_limit(min(image.size),224 * 3.0/4.0)

        z = self.random_zoom_factor(zoom_min,zoom_max)

        image, z = self.zoom(image, x, y, z,return_zoom_factor=True)
        label = label * z

        # other standard transforms
        image = tf.Resize(224)(image)
        image = tf.CenterCrop(224)(image)
        image = tf.RandomHorizontalFlip(p=0.5)(image)
        image = tf.RandomVerticalFlip(p=0.5)(image)

        # COLOR JITTER
        if random.random() > 0.5:
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

        # Gaussian
        image = tf.RandomApply([tf.GaussianBlur(kernel_size=(1, 3), sigma=(1, 2))], p=0.5)(image)

        # finalizing
        image = tf.ToTensor()(image)
        image = tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)

        return image, label

    def val_transform_zoom(self, image, x, y):# becuase this is the only one zooming

        z = 1.0
        image = self.zoom(image, x, y, z)

        image = tf.Resize(224)(image)
        image = tf.CenterCrop(224)(image)
        image = tf.ToTensor()(image)
        image = tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)

        return image

    
    def __getitem__(self, index):
        # importing gan/image
        if self.gan_path is None:
            image_location = self.df[self.image_path].iloc[index]
        elif self.image_path is None:
            image_location = self.df[self.gan_path].iloc[index]
        else:
            if np.random.random() <= 0.5:
                image_location = self.df[self.gan_path].iloc[index]
            else:
                image_location = self.df[self.image_path].iloc[index]
        
        try:
            image = ImageOps.exif_transpose(Image.open(image_location).convert("RGB"))
        except:
            random_idx = np.random.choice(self.df.shape[0])
            with open("none_importable_images.txt", 'a+') as fh:
                fh.write(image_location + ',' + self.df[self.image_path].iloc[random_idx]+ '\n')
            return self.__getitem__(random_idx)


        label = float(self.df[self.label_col].iloc[index])
        label224 = label * (224.0/np.min(image.size)) #standardize label to 224
        
        if self.training_stage == "train":
            image, label224 = self.train_transform_zoom(image, label224, self.df["center_x"].iloc[index], self.df["center_y"].iloc[index]) # for zooming
            #image = self.train_transform(image) # for non-zooming
        elif self.training_stage == "val":
            image = self.val_transform_zoom(image,self.df["center_x"].iloc[index], self.df["center_y"].iloc[index])
        else:
            raise Exception("Training stage not specified")

        return image, label224

class CustomImageDatasetEval(Dataset):

    def __init__(self, df, image_path = "image_path"):
        super().__init__()
        self.df = df
        self.image_path = image_path

    def __len__(self):
        return self.df.shape[0]
    
    def zoom(self, original, x, y, zoom_factor,pixel_zoom=None,return_zoom_factor=False):
        'zoom_factor == zoom_in == 1/zoom_out'

        if pixel_zoom is None:
            side_length = (1.0/zoom_factor) * min(original.size)
        else:
            side_length = pixel_zoom

        left = x - side_length / 2.0
        right = x + side_length / 2.0
        top = y - side_length / 2.0
        bottom = y + side_length / 2.0

        cropped = original.crop((left, top, right, bottom))

        #actual zoom factor, due to rounding deviation
        actual_zoom_factor = float(np.min(original.size)) / float(cropped.size[np.argmin(original.size)])
        
        if not return_zoom_factor:
            return cropped
        else:
            return cropped, actual_zoom_factor
        
    def eval_transform_zoom(self, image, x, y):# becuase this is the only one zooming

        z = 1.0
        image = self.zoom(image, x, y, z) #although z=1 doesn't zoom, this function centers image onto selected point, hence can't remove

        image = tf.Resize(224)(image)
        image = tf.CenterCrop(224)(image)
        image = tf.ToTensor()(image)
        image = tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)

        return image

    def __getitem__(self, index):
        
        # importing gan/image
        image_location = self.df[self.image_path].iloc[index]
        
        try:
            image = ImageOps.exif_transpose(Image.open(image_location).convert("RGB"))
        except:
            random_idx = np.random.choice(self.df.shape[0])
            with open("none_importable_images.txt", 'a+') as fh:
                fh.write(image_location + ',' + self.df[self.image_path].iloc[random_idx]+ '\n')
            return self.__getitem__(random_idx)

        og_wh_min = np.min(image.size)
        image = self.eval_transform_zoom(image,self.df["center_x"].iloc[index], self.df["center_y"].iloc[index])

        return image, image_location, og_wh_min
