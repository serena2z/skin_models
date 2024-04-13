import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error as mse
import argparse


def compute_metrics(df):
    #computing wh_min if it doesn't exist in original df
    df['wh_min'] = (224.0 * (df.pred_pxcm / df.pred_pxcm224)).round().astype(int)
    #computing pxcm224 if it doesn't exist in original df
    df['pxcm224'] = (224.0 / df.wh_min) * df.pxcm

    x = np.log(df.pxcm.values)
    y = np.log(df.pred_pxcm.values)

    log_rmse = np.sqrt(mse(x,y))
    log_r2 = r2_score(x,y)

    print('R2 - log(pxcm) vs log(pxcm)',log_r2)
    print('RMSE - log(pxcm) vs log(pxcm)',log_rmse)


    fig, ax = plt.subplots()
    ax.scatter(x,y)
    ax.set_xlim([np.min(x),np.max(x)])
    ax.set_ylim([np.min(x),np.max(x)])
    ax.set_aspect(1./ax.get_data_ratio(),adjustable='box')
    ax.set_xlabel(r'$log(pxcm_{true})$')
    ax.set_ylabel(r'$log(pxcm_{pred})$')

    # save the figure
    plt.savefig('log_pxcm_vs_log_pxcm.png')

    return log_rmse, log_r2

def main():
    parser = argparse.ArgumentParser(description="Script for evaluating pixel/cm values.")
    parser.add_argument("--input_file", type=str, default="./predictions.txt", help="Path to the dataset", required=True)

    args = parser.parse_args()
    input_file = args.input_file

    df = pd.read_csv(input_file, header=0)

    log_rmse, log_r2 = compute_metrics(df)

if __name__ == "__main__":
    main()


