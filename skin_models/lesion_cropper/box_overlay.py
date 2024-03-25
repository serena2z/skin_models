import cv2
import numpy as np
import pandas as pd
import argparse
import ast

def parse_points(box_points_str):
    points_list = ast.literal_eval(box_points_str)  # Parse string to list
    print('points list: ', points_list)
    ctr = np.array(points_list).reshape((-1,1,2)).astype(np.int32)
    return ctr

def overlay_boxes(df, col_name):
    for index, row in df.iterrows():
        img = cv2.imread(row[col_name])
        # change the name before the .ext to _overlay.ext
        outpath = row['image_path'].replace(".", "_overlay.")

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
    parser.add_argument("--csv_file", type=str, help="Path to the CSV file containing image paths and overlay boxes", required=True, default="predicted_boxes.txt")
    parser.add_argument("--col_name", type=str, help="Name of the column containing image paths", required=True)
    args = parser.parse_args()

    csv_file = args.csv_file
    col_name = args.col_name

    df = pd.read_csv(csv_file, header=0)
    overlay_boxes(df, col_name)

if __name__ == "__main__":
    main()
