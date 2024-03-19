import argparse
import os
import torch
import torchvision.transforms as tf
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torchvision.models.segmentation
from utils import get_image_patches, stitch_patches
from dataloader import RunImageDataset

def run(dataloader, device, model_name, save_dir):
    device = torch.device(device)
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
    model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))
    model = model.to(device)
    model.load_state_dict(torch.load(model_name))
    model.eval()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i, (image, height, width, img_file) in enumerate(tqdm(dataloader)):
        og_shape = image.shape
        image = get_image_patches(image)
        image = image.to(device)

        with torch.no_grad():
            Prd = model(image)['out']
            Prd = stitch_patches(Prd, [og_shape[0], 2, og_shape[2], og_shape[3]])

        for j in range(len(Prd)):
            temp = tf.Resize((int(height[j]), int(width[j])))(Prd[j])
            seg = torch.argmax(temp, 0).cpu().detach().numpy()
            Image.fromarray(seg.astype(np.uint8) * 255).convert('L').save(
                os.path.join(save_dir, os.path.basename(img_file[j]).split('.')[0] + '.png'))

def main():
    parser = argparse.ArgumentParser(description="Run Segmentation on Images")
    parser.add_argument("--img_folder", type=str, required=True, help="Path to the folder containing images")
    parser.add_argument("--model_name", type=str, default="./purple_segmenter.pt", help="Path to the pre-trained model")
    parser.add_argument("--save_dir", type=str, default="./masks", help="Directory to save segmentation masks")
    parser.add_argument("--device", type=str, default="cpu", help="Device for training (cpu or cuda:0)")
    parser.add_argument("--batch_size", type=int, default=11, help="Batch size for dataloader")
    parser.add_argument("--num_workers", type=int, default=12, help="Number of workers for dataloader")

    args = parser.parse_args()

    img_folder = args.img_folder
    batchSize = args.batch_size
    numWorkers = args.num_workers
    model_name = args.model_name
    save_dir = args.save_dir
    device = args.device

    dataset = RunImageDataset(img_folder)
    dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=False, num_workers=numWorkers)
    run(dataloader, device, model_name, save_dir)

if __name__ == "__main__":
    main()