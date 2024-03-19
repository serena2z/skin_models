import argparse
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torchvision.models.segmentation
from dataloader import CustomImageDataset

def evaluate(model_name, dataloader, device):
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
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
    parser.add_argument("--img_folder", type=str, required=True, help="Path to the image folder")
    parser.add_argument("--mask_folder", type=str, required=True, help="Path to the mask folder")
    parser.add_argument("--model_path", type=str, default="./skin_segmenter.pt", required=True, help="Path to the trained model")
    parser.add_argument("--device", type=str, default="cpu", help="Device for training (cpu or cuda:0)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for dataloader")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for dataloader")

    args = parser.parse_args()

    img_folder = args.img_folder
    mask_folder = args.mask_folder
    model_path = args.model_path
    device = args.device
    batch_size = args.batch_size
    num_workers = args.num_workers

    dataset = CustomImageDataset(img_folder, mask_folder, transforms=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    evaluate(model_path, dataloader, device)

if __name__ == "__main__":
    main()
