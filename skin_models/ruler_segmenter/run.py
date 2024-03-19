import argparse
import numpy as np
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as tf
import torchvision.models.segmentation
from PIL import Image
from dataloader import RunImageDataset

def run(dataloader, model_name, save_dir, device):
    device = torch.device(device)
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
    model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))
    model = model.to(device) 
    model.load_state_dict(torch.load(model_name, map_location=torch.device(device))) 
    model.eval() 

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

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
    parser.add_argument("--img_folder", type=str, required=True, help="Path to the image folder")
    parser.add_argument("--model_path", type=str, default="./ruler_segmenter.pt", help="Path to the pre-trained model")
    parser.add_argument("--save_dir", type=str, default="./masks", help="Directory to save segmentation masks")
    parser.add_argument("--device", type=str, default="cpu", help="Device for training (cpu or cuda:0)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for dataloader")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for dataloader")

    args = parser.parse_args()

    img_folder = args.img_folder
    model_path = args.model_path
    save_dir = args.save_dir
    device = args.device
    batch_size = args.batch_size
    num_workers = args.num_workers

    dataset = RunImageDataset(img_folder)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    run(dataloader, model_path, save_dir, device)
            
if __name__ == "__main__":
    main()
