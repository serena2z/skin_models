import cv2
import numpy as np
import pandas as pd
import argparse
import ast
import os

def parse_points(box_points_str):
    points_list = ast.literal_eval(box_points_str)  # Parse string to list
    print('points list: ', points_list)
    ctr = np.array(points_list).reshape((-1,1,2)).astype(np.int32)
    return ctr

def overlay_boxes(df, output_dir):
    for index, row in df.iterrows():
        img = cv2.imread(row['image_path'])
        if os.path.exists(output_dir) == False:
            os.makedirs(output_dir)
        # create a new folder path using output_dir and the image path
        outpath = output_dir + '/' + row['image_path'].split('/')[-1]

        box_points = parse_points(row['box_points'])
        box_points_square = parse_points(row['box_points_square'])

        # Draw the rotated box
        thickness1 = int(np.ceil(np.average(img.shape[0:2]) / 512.0 * 8))
        thickness2 = int(np.ceil(np.average(img.shape[0:2]) / 512.0 * 3))

        out_img = cv2.drawContours(img.copy(), [box_points_square], 0, (255, 255, 0), thickness = thickness1)
        out_img = cv2.drawContours(out_img.copy(), [box_points], 0, (0, 0, 255), thickness = thickness2)

        # Resize the image such that the smaller edge is 512
        height, width = out_img.shape[:2]
        scale = 512 / min(height, width)
        new_width, new_height = int(width * scale), int(height * scale)
        resized_out = cv2.resize(out_img, (new_width, new_height))

        # Save the image with overlaid boxes
        cv2.imwrite(outpath, resized_out)

def main():
    parser = argparse.ArgumentParser(description="Overlay predicted boxes onto images.")
    parser.add_argument("--txt_file", type=str, help="Path to the text file containing image paths and overlay boxes", required=True, default="predicted_boxes.txt")
    parser.add_argument("--output_dir", type=str, help="Name of the output folder containing image visualizations", required=True, default="output")
    args = parser.parse_args()

    csv_file = args.txt_file
    output_dir = args.output_dir

    df = pd.read_csv(csv_file, header=0)
    overlay_boxes(df, output_dir)

if __name__ == "__main__":
    main()
