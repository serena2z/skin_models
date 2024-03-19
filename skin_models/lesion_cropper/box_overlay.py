import cv2
import numpy as np
import pandas as pd
import argparse

def overlay_boxes(df):
    for index, row in df.iterrows():
        img = cv2.imread(row['image_path'])
        outpath = row['out_path']

        # Read the highest_score_box from the CSV file
        box_points = row['highest_score_box']
        box_points_square = row['highest_score_box_square']

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
    args = parser.parse_args()

    csv_file = args.csv_file

    df = pd.read_csv(csv_file)
    overlay_boxes(df)

if __name__ == "__main__":
    main()
