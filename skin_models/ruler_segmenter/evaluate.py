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
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False)
    model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))
    model.load_state_dict(torch.load(model_name, map_location=torch.device(device))) 
    model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    dice_nan_count = 0
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader)):
            images, masks = data
            masks = masks.squeeze(1).long()
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)['out']
            loss = loss_fn(outputs, masks)
            running_loss += loss.item() * images.size(0)
            for i in range(outputs.shape[0]):
                seg = torch.argmax(outputs[i], axis=0).cpu().detach().numpy()
                gt = masks[i].detach().cpu().numpy()
                dice = dice_score(seg, gt)
                if ~np.isnan(dice):
                    running_dice += dice
                else:
                    dice_nan_count += 1
    avg_loss = running_loss / len(dataloader.dataset)
    avg_dice = running_dice / (len(dataloader.dataset) - dice_nan_count)
    print('Test Loss: {}, Test Dice Score: {}'.format(avg_loss, avg_dice))
    return avg_loss, avg_dice

def dice_score(seg, gt, k=1):
    dicey = np.sum(seg[gt==k]) * 2.0 / (np.sum(seg) + np.sum(gt)) 
    return dicey

def main():
    parser = argparse.ArgumentParser(description="Semantic Segmentation Evaluation with DeepLabV3")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to the test dataset")
    parser.add_argument("--col_name", type=str, default="file_name", help="Name of the column containing the image file names")
    parser.add_argument("--mask_col_name", type=str, default="mask_file_name", help="Name of the column containing the mask file names")
    parser.add_argument("--model_path", type=str, default="./ruler_segmenter.pt", required=True, help="Path to the trained model")
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
