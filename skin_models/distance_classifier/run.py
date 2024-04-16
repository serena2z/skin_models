import argparse
import pandas as pd 
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error as mse
from dataloader import CustomImageDatasetEval
import ast
import numpy as np


def run(dataloader, model_name, save_path, device):
    device = torch.device(device)
    model = torchvision.models.resnet50(pretrained=True) 
    model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=1, bias=True)
    model = model.to(device) 
    model.load_state_dict(torch.load(model_name, map_location=torch.device(device))) 
    model.eval() 

    df_save = pd.DataFrame()

    for i, (images, image_path, og_wh_min) in enumerate(tqdm(dataloader)):

        images = images.to(device)
        
        with torch.no_grad():
            Pred_log = model(images) 
            Pred = torch.exp(Pred_log) - 1

        pxcm224 = Pred.detach().cpu().squeeze().numpy()
        og_wh_min = og_wh_min.detach().numpy()
        pxcm = (og_wh_min/224.0) * pxcm224


        image_path = np.atleast_1d(image_path)
        pxcm224 = np.atleast_1d(pxcm224)
        pxcm = np.atleast_1d(pxcm)

        for im, p224, p in zip(image_path, pxcm224, pxcm):
            nrow = len(df_save)
            df_save.loc[nrow,'image_path'] = im
            df_save.loc[nrow,'pred_pxcm224'] = p224
            df_save.loc[nrow,'pred_pxcm'] = p

    df_save.to_csv(save_path)
    

def main():
    parser = argparse.ArgumentParser(description="Script for predicting and evaluating pixel/cm values.")
    parser.add_argument("--input_file", type=str, default="./box_predictions.txt", help="Path to the dataset", required=True)
    parser.add_argument("--model_path", type=str, default="./model.torch", help="Path to the trained model", required=True)
    parser.add_argument("--save_path", type=str, default="./distance_predictions.csv", help="Path to save predictions")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run the model on")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for data loading")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for data loading")
    
    args = parser.parse_args()

    batch_size = args.batch_size
    num_workers = args.num_workers
    model_path = args.model_path
    save_path = args.save_path
    input_file = args.input_file
    device = args.device

    df = pd.read_csv(input_file, header=0)
    
    # comment out if you don't need to extract center_x and center_y from box_center
    # df["center_x"] = None
    # df["center_y"] = None

    # for index, row in df.iterrows():
    #     highest_score_box = ast.literal_eval(row["box_center"])
    #     df.loc[index, "center_x"] = (highest_score_box[0])
    #     df.loc[index, "center_y"] = (highest_score_box[1])
    
    dataset = CustomImageDatasetEval(df)
    dataloader = DataLoader(dataset, batch_size= batch_size, shuffle=False, num_workers = num_workers)

    run(dataloader, model_path, save_path, device)

if __name__ == "__main__":
    main()



