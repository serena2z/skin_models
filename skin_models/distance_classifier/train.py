# %%
import pandas as pd 
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tf
import torchvision.transforms.functional as F
import numpy as np
import torchvision
import os
from tqdm import tqdm
import sys
import sklearn.metrics
from dataloader import CustomImageDataset
import argparse


def train(dataloader, save_dir, device, epochs, val_dataloader=None):
    device = torch.device(device)
    model = torchvision.models.resnet50(pretrained=True)
    model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=1, bias=True)
    model = model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    best_val_loss = float('inf')
    best_val_r2 = 0.0
    best_val_rmse = float('inf')
    best_val_rmse_log = float('inf')

    for epoch in range(epochs):
        print('EPOCH {}:'.format(epoch))
        model.train()
        avg_loss = train_one_epoch(dataloader, model, optimizer, device, scheduler)
        print('Training Loss: {}'.format(avg_loss))
        model.train(False)
        
        if val_dataloader:
            val_loss, val_rmse, val_r2, val_rmse_log = validate(val_dataloader, model, device)
            if val_loss < best_val_loss:
                print("saving best loss model #" + str(epoch), flush = True)
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_dir + "/best_val_loss_" + str(epoch) + "_" + str(best_val_loss) + ".torch")

            if val_rmse < best_val_rmse:
                print("saving best rmse model #" + str(epoch), flush = True)
                best_val_rmse = val_rmse
                torch.save(model.state_dict(), save_dir + "/best_val_rmse_" + str(epoch) + "_" + str(best_val_rmse) + ".torch")

            if val_rmse_log < best_val_rmse_log:
                print("saving best rmse model #" + str(epoch), flush = True)
                best_val_rmse_log = val_rmse_log
                torch.save(model.state_dict(), save_dir + "/best_val_rmse_log_" + str(epoch) + "_" + str(best_val_rmse_log) + ".torch")
        else:
            model_path = save_dir + '/model_' + str(epoch) + '.pt'
            torch.save(model.state_dict(), model_path)


def train_one_epoch(dataloader, model, optimizer, device, scheduler):
    running_loss = 0.0
    for images, pxcm in enumerate(tqdm(dataloader)):
        images = images.to(device)
        pxcm = pxcm.float().unsqueeze(1).to(device)

        Pred = model(images)
        model.zero_grad()
        Loss = torch.abs(Pred - pxcm).mean()
        Loss.backward()
        optimizer.step()

        running_loss += Loss.item() * images.size(0)
    scheduler.step()
    return running_loss / len(dataloader)


def validate(val_dataloader, model, device):
    model.eval()

    running_val_loss = 0.0
    running_val_r2 = 0.0
    running_val_rmse = 0.0
    running_val_rmse_log = 0.0

    with torch.no_grad():
        for images, pxcm in enumerate(tqdm(val_dataloader)):
            images = images.to(device)
            pxcm = pxcm.float().unsqueeze(1).to(device)
            pxcm_log = torch.log(pxcm.float() + 1).unsqueeze(1).to(device)

            Pred_log = model(images) 
            val_loss = torch.abs(Pred_log - pxcm_log).mean()

            Pred = torch.exp(Pred_log) - 1

            val_mse = sklearn.metrics.mean_squared_error(Pred.detach().cpu().numpy(), pxcm.detach().cpu().numpy())
            val_rmse = np.sqrt(val_mse)
            val_r2 = sklearn.metrics.r2_score(Pred.detach().cpu().numpy(), pxcm.detach().cpu().numpy())

            val_mse_log  = sklearn.metrics.mean_squared_error(Pred_log.detach().cpu().numpy().squeeze(), pxcm_log.detach().cpu().numpy().squeeze()) 
            val_rmse_log = np.sqrt(val_mse_log)
            
            running_val_loss += val_loss.item() * images.size(0)
            running_val_r2 += val_r2 * images.size(0)
            running_val_rmse += val_rmse * images.size(0)

            running_val_rmse_log += val_rmse_log * images.size(0)


    val_loss = running_val_loss / len(val_dataloader)
    val_rmse = running_val_rmse / len(val_dataloader)
    val_r2 = running_val_r2 / len(val_dataloader)
    val_rmse_log = running_val_rmse_log / len(val_dataloader)
    print('Validation Loss: {}, Validation RMSE: {}, Validation R2: {}, Validation RMSE Log: {}'.format(val_loss, val_rmse, val_r2, val_rmse_log))
    return val_loss, val_rmse, val_r2, val_rmse_log


def main():
    parser = argparse.ArgumentParser(description="Script for training pixel/cm values.")
    parser.add_argument("--batch_size", type=int, default=192, help="Batch size for data loading", required=True)
    parser.add_argument("--num_workers", type=int, default=24, help="Number of workers for data loading", required=True)
    parser.add_argument("--save_dir", type=str, default="./models", help="Path to save the trained model")
    parser.add_argument("--epochs", type=int, default=250, help="Number of epochs to train", required=True)
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the model on", required=True)
    parser.add_argument("--train_file", type=str, default="./data.csv", help="Path to the dataset", required=True)

    # Add validation arguments
    parser.add_argument("--val", action="store_true", help="Use validation dataset")
    parser.add_argument("--val_file", type=str, default="./val_images", help="Path to the validation image folder")
    
    args = parser.parse_args()

    batch_size = args.batch_size
    num_workers = args.num_workers
    save_dir = args.save_dir
    epochs = args.epochs
    device = args.device
    train_file = args.train_file
    val = args.val
    val_file = args.val_file

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    df = pd.read_csv(train_file, header=0)
    dataset = CustomImageDataset(df, training_stage="train")
    dataloader = DataLoader(dataset, batch_size= batch_size, shuffle=True, num_workers = num_workers)

    if val:
        df_val = pd.read_csv(val_file, header=0)
        val_dataset = CustomImageDataset(df_val, training_stage="val")
        val_dataloader = DataLoader(val_dataset, batch_size= batch_size, shuffle=False, num_workers = num_workers)
        train(dataloader, save_dir, device, epochs, val_dataloader)
    else:
        train(dataloader, save_dir, device, epochs)

if __name__ == "__main__":
    main()
