
import argparse
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tf
import torchvision.models.segmentation
import torchvision.transforms.functional as F
from PIL import Image, ImageOps

WIDTH = HEIGHT = 512

class CustomImageDataset(Dataset):
    def __init__(self, df, img_path, transforms):
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
            with open('none_importable_images.txt','a+') as fh:
                fh.write(img_location +', ' + self.df[self.img_path].iloc[random_idx] + '\n')                
            return self.__getitem__(random_idx)
        
        image_width, image_height = image.size

        if self.transforms:
            image = self.transform(image)
        else: 
            image = tf.Resize((HEIGHT, WIDTH))(image)
            image = tf.ToTensor()(image)
            image = tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)

        return image, image_height, image_width, img_location
    
    # def transform(self, image: Image.Image) -> Image.Image:
    #     """Applies data augmentation transforms to the image."""
    #     transform_ops = tf.Compose([
    #         tf.Resize((HEIGHT, WIDTH)),
    #         tf.RandomHorizontalFlip(p=0.5),
    #         tf.RandomVerticalFlip(p=0.5),
    #         tf.RandomAffine(degrees=[-30, 30], translate=(0.3, 0.3), scale=(0.75, 1.25), shear=[-10, 10, -10, 10], interpolation=Image.BILINEAR),
    #         tf.ColorJitter(brightness=(0.9, 1.1), contrast=(0.9, 1.1), saturation=(0.9, 1.1), hue=(-0.1, 0.1)),
    #         tf.ToTensor(),
    #         tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #     ])
    #     return transform_ops(image)


def run(dataloader, model_name, save_dir, device):
    device = torch.device(device)
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
    model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))
    model = model.to(device) 
    model.load_state_dict(torch.load(model_name, map_location=torch.device(device))) 
    model.eval() 

    for i, (image, height, width, img_file) in enumerate(tqdm(dataloader)):
        image = image.to(device)
        with torch.no_grad():
            Prd = model(image)['out']
        for j in range(len(Prd)):
            temp = tf.Resize((int(height[j]),int(width[j])))(Prd[j])
            seg = torch.argmax(temp, 0).cpu().detach().numpy()
            Image.fromarray(seg.astype(np.uint8)*255).convert('L').save(os.path.join(save_dir, os.path.basename(img_file[j]).split('.')[0] + '.png'))

def main():
    parser = argparse.ArgumentParser(description="Semantic Segmentation with DeepLabV3")
    parser.add_argument("--data_path", type=str, default="./data.csv", help="Path to the dataset")
    parser.add_argument("--col_name", type=str, default="file_name", help="Name of the column containing the image file names")
    parser.add_argument("--model_path", type=str, default="./skin_segmenter.pt", help="Path to the pre-trained model")
    parser.add_argument("--save_dir", type=str, default="./masks", help="Directory to save segmentation masks")
    # parser.add_argument("-transforms", action="store_true", help="Apply data augmentation")
    parser.add_argument("--device", type=str, default="cpu", help="Device for training (cpu or cuda:0)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for dataloader")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for dataloader")

    args = parser.parse_args()

    data_path = args.data_path
    col_name = args.col_name
    model_path = args.model_path
    save_dir = args.save_dir
    transforms = args.transforms
    device = args.device
    batch_size = args.batch_size
    num_workers = args.num_workers

    dataset = CustomImageDataset(pd.read_csv(data_path), col_name, transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    run(dataloader, model_path, save_dir, device)
            
if __name__ == "__main__":
    main()