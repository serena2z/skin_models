import argparse
import numpy as np
import pandas as pd
import torch
import torchvision.models as models
import torchvision.transforms as tf
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
import csv
from dataloader import RunImageDataset


def run(dataloader, model_name, save_file, device):
    device = torch.device(device)
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, out_features=11, bias=True)
    model = model.to(device)  # Set model to GPU or CPU
    model.load_state_dict(torch.load(model_name, map_location=torch.device(device))) # Load trained model
    model.eval()

    f = open(save_file, 'w+')
    writer = csv.writer(f)
    writer.writerow(['file_name', 'prediction', 'probabilities'])

    with torch.no_grad():
        for i, (inputs, paths) in enumerate(tqdm(dataloader)):

            inputs = inputs.to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs,1)
            _, preds = torch.max(outputs, 1)
            
            # save the paths and the predictions in a csv file

            for i in range(len(paths)):
                preds_list = preds.detach().cpu().numpy().tolist()
                probs_list = probs.detach().cpu().numpy().tolist()
                writer.writerow([paths[i], preds_list[i], probs_list[i]])

    f.close()


def main():
    parser = argparse.ArgumentParser(description="Semantic Segmentation with DeepLabV3")
    parser.add_argument("--data_path", type=str, default="./data.csv", help="Path to the dataset")
    parser.add_argument("--col_name", type=str, default="file_name", help="Name of the column containing the image file names")
    parser.add_argument("--model_path", type=str, default="./body_classifier.pt", help="Path to the pre-trained model")
    parser.add_argument("--save_file", type=str, default="./results.csv", help="Directory to save the predictions and probabilities")
    parser.add_argument("--device", type=str, default="cpu", help="Device for training (cpu or cuda:0)")
    parser.add_argument("--batch_size", type=int, default=14, help="Batch size for dataloader")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for dataloader")

    args = parser.parse_args()
    print(args.transforms)
    data_path = args.data_path
    col_name = args.col_name
    model_path = args.model_path
    save_file = args.save_file
    transforms = args.transforms
    device = args.device
    batch_size = args.batch_size
    num_workers = args.num_workers

    dataset = RunImageDataset(pd.read_csv(data_path), col_name)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    run(dataloader, model_path, save_file, device)

if __name__ == "__main__":
    main()
