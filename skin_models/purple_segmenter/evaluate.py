import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tf
import torchvision.models.segmentation
import torchvision.transforms.functional as F
from PIL import Image, ImageOps

WIDTH = HEIGHT = 512
patch_size = (224,224)

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

class CustomImageDataset(Dataset):
    def __init__(self, df, img_path, mask_path):
        super().__init__()
        self.df = df
        self.img_path = img_path
        self.mask_path = mask_path

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

        image = tf.Resize((HEIGHT, WIDTH))(image)
        image = tf.ToTensor()(image)
        image = tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
        mask = tf.Resize((HEIGHT, WIDTH), tf.InterpolationMode.NEAREST)(mask)
        mask_np = np.array(mask, dtype=np.uint8)
        mask_np = mask_np / np.max(np.abs(mask_np)) if ((len(np.unique(mask_np)) != 1) or (np.sum(mask_np) != 0)) else mask_np
        mask_np = mask_np.astype(np.float32)
        mask = tf.ToTensor()(mask_np)

        return image, mask, image_height, image_width, img_location

def evaluate(model_name, dataloader, device):
    device = torch.device(device)
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
    model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))
    model.load_state_dict(torch.load(model_name, map_location=torch.device(device))) 
    model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    model.eval()
    running_vloss = 0.0
    vdice = []
    vdice_f = [] #full image dice
    with torch.no_grad():
        for i, vdata in enumerate(tqdm(dataloader)):
            vimages, vmasks = vdata
            shape_f = vimages.shape
            vmasks_f = vmasks.clone().detach().squeeze(1).long()
            vimages = get_image_patches(vimages, patch_size)
            vmasks = get_image_patches(vmasks,patch_size)

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
    print('Test loss {} , patch dice {} full image dice {}'.format(avg_vloss, avg_vdice, avg_vdice_f))
    return avg_vloss, avg_vdice, avg_vdice_f

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

def main():
    parser = argparse.ArgumentParser(description="Semantic Segmentation Evaluation with DeepLabV3")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to the test dataset")
    parser.add_argument("--col_name", type=str, default="file_name", help="Name of the column containing the image file names")
    parser.add_argument("--mask_col_name", type=str, default="mask_file_name", help="Name of the column containing the mask file names")
    parser.add_argument("--model_path", type=str, default="./skin_segmenter.pt", required=True, help="Path to the trained model")
    parser.add_argument("--device", type=str, default="cpu", help="Device for training (cpu or cuda:0)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for dataloader")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for dataloader")

    args = parser.parse_args()

    test_data_path = args.test_data_path
    col_name = args.col_name
    mask_col_name = args.mask_col_name
    model_path = args.model_path
    device = args.device
    batch_size = args.batch_size
    num_workers = args.num_workers

    dataset = CustomImageDataset(pd.read_csv(test_data_path), col_name, mask_col_name, transforms=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    evaluate(model_path, dataloader, device)

if __name__ == "__main__":
    main()
