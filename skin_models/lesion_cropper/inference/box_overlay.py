import cv2
import numpy as np
import pandas as pd
import argparse
import ast
import os

def parse_points(box_points_str):
    points_list = ast.literal_eval(box_points_str)  # Parse string to list
    ctr = np.array(points_list).reshape((-1,1,2)).astype(np.int32)
    return ctr

def overlay_boxes(df, output_dir, thickness=2, resize=False):
    for index, row in df.iterrows():
        img = cv2.imread(row['image_path'])
        if os.path.exists(output_dir) == False:
            os.makedirs(output_dir)
        # create a new folder path using output_dir and the image path
        outpath = output_dir + '/' + row['image_path'].split('/')[-1]

        box_points = parse_points(row['box_points'])
        box_points_square = parse_points(row['box_points_square'])

        # Draw the rotated box
        thickness1 = int(np.ceil(np.average(img.shape[0:2]) / 512.0 * (thickness * 3)))
        thickness2 = int(np.ceil(np.average(img.shape[0:2]) / 512.0 * thickness))

        out_img = cv2.drawContours(img.copy(), [box_points_square], 0, (255, 255, 0), thickness = thickness1)
        out_img = cv2.drawContours(out_img.copy(), [box_points], 0, (0, 0, 255), thickness = thickness2)

        # Resize the image such that the smaller edge is 512
        if resize:
            height, width = out_img.shape[:2]
            scale = 512 / min(height, width)
            new_width, new_height = int(width * scale), int(height * scale)
            resized_out = cv2.resize(out_img, (new_width, new_height))
        else:
            resized_out = out_img

        cv2.imwrite(outpath, resized_out)

def main():
    parser = argparse.ArgumentParser(description="Overlay predicted boxes onto images.")
    parser.add_argument("--input_file", type=str, help="Path to the text file containing image paths and overlay boxes", required=True, default="predicted_boxes.txt")
    parser.add_argument("--save_dir", type=str, help="Name of the output folder containing image visualizations", required=True, default="output")
    parser.add_argument("--thickness", type=int, help="Thickness to use to draw the boxes", required=False, default=2)
    parser.add_argument("--resize", action="store_true", help="Resizes the smaller side of the image to 512 while keeping aspect ratio constant, used to reduce image storage", required=False)
    args = parser.parse_args()

    input_file = args.input_file
    save_dir = args.save_dir
    thickness = args.thickness
    resize = args.resize

    df = pd.read_csv(input_file, header=0)
    overlay_boxes(df, save_dir, thickness, resize)

if __name__ == "__main__":
    main()
