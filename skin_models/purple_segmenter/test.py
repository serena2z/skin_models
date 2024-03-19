import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torchvision.models.segmentation
from utils import get_image_patches, stitch_patches, dice_score
from dataloader import CustomImageDataset

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
            vimages = get_image_patches(vimages)
            vmasks = get_image_patches(vmasks)

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
            voutputs_f = stitch_patches(voutputs,[shape_f[0],2,shape_f[2],shape_f[3]])
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

def main():
    parser = argparse.ArgumentParser(description="Semantic Segmentation Evaluation with DeepLabV3")
    parser.add_argument("--img_folder", type=str, required=True, help="Path to the folder containing images")
    parser.add_argument("--mask_folder", type=str, required=True, help="Path to the folder containing masks")
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
