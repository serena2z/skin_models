from PIL import Image, ImageOps, ImageTransform
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from torchvision import transforms as tf
import argparse
import ast

def crop_image_with_padding(pil_img, box_cx, box_cy, box_w, box_h, angle, pad_color=(255, 255, 255)):
    """
    Crop the image with padding.

    Args:
        pil_img (PIL.Image): Input image.
        box_cx (float): X-coordinate of the center of the box.
        box_cy (float): Y-coordinate of the center of the box.
        box_w (float): Width of the box.
        box_h (float): Height of the box.
        angle (float): Rotation angle of the box.
        pad_color (tuple): Padding color.

    Returns:
        PIL.Image: Cropped and padded image.
    """
    img = pil_img

    # Run model
    box_points = cv2.boxPoints(((box_cx, box_cy), (box_w, box_h), angle))

    # Get min max points
    x_min, x_max = [box_points[:, 0].min(), box_points[:, 0].max()]
    y_min, y_max = [box_points[:, 1].min(), box_points[:, 1].max()]

    # Get image size
    w, h = img.size

    # Calculate padding
    pad_left = max(0, -int(np.floor(x_min)))
    pad_right = max(0, int(np.ceil(x_max - w)))
    pad_top = max(0, -int(np.floor(y_min)))
    pad_bot = max(0, int(np.ceil(y_max - h)))

    # Update box points and center points
    box_cx2 = box_cx + pad_left
    box_cy2 = box_cy + pad_top
    box_points2 = box_points + [pad_left, pad_top]

    # Make padded image
    if np.any(np.array([pad_left, pad_right, pad_top, pad_bot]) != 0):
        img2 = add_padding(img.copy(), pad_left, pad_right, pad_top, pad_bot, pad_color)
    else:
        img2 = img.copy()

    # Apply crop of rotated
    transform = np.round(box_points2[[1, 0, 3, 2], :]).flatten()
    crop_img = img2.transform((int(np.round(box_w)), int(np.round(box_h))), 
                              ImageTransform.QuadTransform(transform), 
                              resample=Image.Resampling.BICUBIC)

    return crop_img


def add_padding(pil_img, left, right, top, bottom, color):
    """
    Add padding to the image.

    Args:
        pil_img (PIL.Image): Input image.
        left (int): Left padding.
        right (int): Right padding.
        top (int): Top padding.
        bottom (int): Bottom padding.
        color (tuple): Padding color.

    Returns:
        PIL.Image: Padded image.
    """
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


def main():
    parser = argparse.ArgumentParser(description='Image Cropping and Resizing')
    parser.add_argument('--input_file', type=str, default='box_predictions.txt',
                        help='Path to the input file containing predicted boxes', required=True)
    parser.add_argument('--save_dir', type=str, default='./skin_models/lesion_cropper/samples',
                        help='Directory to save the cropped and resized images', required=True)
    args = parser.parse_args()

    input_file = args.input_file
    save_dir = args.save_dir

    df = pd.read_csv(input_file, header=0)

    # rewrite the image paths using the save_dir
    df['cropped_image_path'] = df['image_path'].map(lambda x: os.path.join(save_dir, 'images_detectron2box', os.path.splitext(os.path.basename(x))[0] + '.png'))
    df['cropped_image_square_path'] = df['image_path'].map(lambda x: os.path.join(save_dir, 'images_detectron2box_sq', os.path.splitext(os.path.basename(x))[0] + '.png'))

    # Create directories if not exist
    for subdir in ['images_detectron2box', 'images_detectron2box_sq']:
        os.makedirs(os.path.join(save_dir, subdir), exist_ok=True)

    for idx in tqdm(df.index.tolist()):
        img_loc = df.loc[idx, 'image_path']
        box_cx, box_cy, box_w, box_h, angle = df.loc[idx, 'center_x'], df.loc[idx, 'center_y'], df.loc[idx, 'width'], df.loc[idx, 'height'], df.loc[idx, 'angle']

        img = ImageOps.exif_transpose(Image.open(img_loc)).convert('RGB')

        # Ensure that box width and height are at least 1 pixel
        box_w = max(1, box_w)
        box_h = max(1, box_h)

        # Crop box
        crop_img = crop_image_with_padding(img, box_cx, box_cy, box_w, box_h, angle)
        crop_img.save(df.loc[idx, 'cropped_image_path'])

        # Square crop box
        box_avg_wh = np.average([box_w, box_h])
        crop_img = crop_image_with_padding(img, box_cx, box_cy, box_avg_wh, box_avg_wh, angle)
        crop_img.save(df.loc[idx, 'cropped_image_square_path'])

if __name__ == "__main__":
    main()

