import argparse
import os
import torch
import torchvision.transforms as tf
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
import torchvision.models.segmentation
import torchvision.transforms.functional as F
from datetime import datetime

WIDTH=HEIGHT=224*5
patch_size = (224,224)

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

class CustomImageDataset(Dataset):
    def __init__(self, df, img_path='file_name', label_col='image_type_enc', transforms=True):
        super().__init__()
        self.df = df
        self.img_path = img_path
        self.transforms = transforms

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        img_location = self.df[self.img_path].iloc[index]
        try:
            image = ImageOps.exif_transpose(Image.open(img_location).convert('RGB'))
        except:
            random_idx = np.random.choice(self.df.shape[0])
            with open('none_importable_images.txt', 'a+') as fh:
                fh.write(img_location + ', ' + self.df[self.img_path].iloc[random_idx] + '\n')
            return self.__getitem__(random_idx)
        image_width, image_height = image.size

        image = tf.Resize((HEIGHT, WIDTH))(image)
        image = tf.ToTensor()(image)
        image = tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)

        return image, image_height, image_width, img_location

def run(dataloader, device, model_name, save_dir):
    device = torch.device(device)
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
    model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))
    model = model.to(device)
    model.load_state_dict(torch.load(model_name))
    loss_fn = torch.nn.CrossEntropyLoss()
    model.eval()

    for i, (image, height, width, img_file) in enumerate(tqdm(dataloader)):
        og_shape = image.shape
        image = get_image_patches(image, patch_size)
        image = image.to(device)

        with torch.no_grad():
            Prd = model(image)['out']
            Prd = stitch_patches(Prd, [og_shape[0], 2, og_shape[2], og_shape[3]], patch_size)

        for j in range(len(Prd)):
            temp = tf.Resize((int(height[j]), int(width[j])))(Prd[j])
            seg = torch.argmax(temp, 0).cpu().detach().numpy()
            Image.fromarray(seg.astype(np.uint8) * 255).convert('L').save(
                os.path.join(save_dir, os.path.basename(img_file[j]).split('.')[0] + '.png'))

def main(args):
    parser = argparse.ArgumentParser(description="Run Segmentation on Images")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data file (CSV)")
    parser.add_argument("--col_name", type=str, default="file_name", help="Name of the column containing the image file names")
    parser.add_argument("--model_path", type=str, default="./ruler_segmenter.pt", help="Path to the pre-trained model")
    parser.add_argument("--save_dir", type=str, default="./masks", help="Directory to save segmentation masks")
    parser.add_argument("--device", type=str, default="cpu", help="Device for training (cpu or cuda:0)")
    parser.add_argument("--batch_size", type=int, default=11, help="Batch size for dataloader")
    parser.add_argument("--num_workers", type=int, default=12, help="Number of workers for dataloader")

    args = parser.parse_args()

    data_path = args.data_path
    col_name = args.col_name
    batchSize = args.batch_size
    numWorkers = args.num_workers
    model_name = args.model_name
    save_dir = args.save_dir
    device = args.device

    dataset = CustomImageDataset(pd.read_csv(data_path), col_name)
    dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=False, num_workers=numWorkers)
    run(dataloader, device, model_name, save_dir)

if __name__ == "__main__":
    main()
