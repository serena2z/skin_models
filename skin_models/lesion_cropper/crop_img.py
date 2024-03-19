import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageTransform
from torchvision import transforms as tf
from tqdm import tqdm
import argparse

def crop_image_wpadding(pil_img, box_cx, box_cy, box_w, box_h, angle, pad_color=(255,255,255)):
    img = pil_img
    box_points = cv2.boxPoints(((box_cx, box_cy), (box_w, box_h), angle))

    x_min, x_max = [box_points[:,0].min(), box_points[:,0].max()]
    y_min, y_max = [box_points[:,1].min(), box_points[:,1].max()] 

    w, h = img.size

    pad_left = 0 if x_min > 0 else abs(int(np.floor(x_min)))
    pad_right = 0 if x_max <= w else int(np.ceil(x_max - w))
    pad_top = 0 if y_min > 0 else abs(int(np.floor(y_min)))
    pad_bot = 0 if y_max <= h else int(np.ceil(y_max - h))

    box_points2 = box_points.copy()
    box_points2[:,0] = box_points2[:,0] + pad_left
    box_points2[:,1] = box_points2[:,1] + pad_top

    if np.any(np.array([pad_left, pad_right, pad_top, pad_bot]) != 0):
        img2 = add_padding(img.copy(), pad_left, pad_right, pad_top, pad_bot, pad_color)
    else:
        img2 = img.copy()

    transform = np.round(box_points2[[1,0,3,2], :]).flatten()
    crop_img = img2.transform((np.round(box_w).astype(int), np.round(box_h).astype(int)), ImageTransform.QuadTransform(transform), resample=Image.Resampling.BICUBIC)
    return crop_img

def add_padding(pil_img, left, right, top, bottom, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

def crop_and_save_images(img, box_cx, box_cy, box_w, box_h, df, idx, zoom_factors):
    for zoom_factor in zoom_factors:
        box_avg_wh = np.average([box_w,box_h])
        orig_zoom_box_wh = int(np.round(zoom_factor * box_avg_wh))
        new_box_wh = 1024
        scale_factor = float(new_box_wh) / orig_zoom_box_wh
        new_box_cx = scale_factor * box_cx
        new_box_cy = scale_factor * box_cy
        new_w = int(np.round(scale_factor * img.size[0]))
        new_h = int(np.round(scale_factor * img.size[1]))
        new_angle = 0

        new_img = tf.Resize((new_h, new_w), interpolation=tf.InterpolationMode.BICUBIC)(img)
        crop_img = crop_image_wpadding(new_img, new_box_cx, new_box_cy, new_box_wh, new_box_wh, new_angle)
        zoom_str = str(zoom_factor).replace(".", "p")
        path = df.columns[-1] + f'_zoomout{zoom_str}_1024_path'
        crop_img.save(df.loc[idx, path])

def main(args):
    # Load dataframe
    df = pd.read_csv(args.input_file, header=None, converters={2: pd.eval})
    df['basename_noext'] = df[0].map(lambda x: os.path.splitext(os.path.basename(x))[0])

    df['cropped_image_path'] = df[0].map(lambda x: os.path.split(os.path.split(x)[0])[0] + '/images_detectron2box/' + os.path.splitext(os.path.basename(x))[0] + '.png')
    # Add paths for other cropped images
    df['cropped_image_square_path'] = df[0].map(lambda x: os.path.split(os.path.split(x)[0])[0] + '/images_detectron2box_sq/' + os.path.splitext(os.path.basename(x))[0] + '.png')
    df['cropped_image_zoomout2_1024_path'] = df[0].map(lambda x: os.path.split(os.path.split(x)[0])[0] + '/images_detectron2box_zoomout2_1024/' + os.path.splitext(os.path.basename(x))[0] + '.png')
    df['cropped_image_zoomout2p5_1024_path'] = df[0].map(lambda x: os.path.split(os.path.split(x)[0])[0] + '/images_detectron2box_zoomout2p5_1024/' + os.path.splitext(os.path.basename(x))[0] + '.png')
    df['cropped_image_zoomout3_1024_path'] = df[0].map(lambda x: os.path.split(os.path.split(x)[0])[0] + '/images_detectron2box_zoomout3_1024/' + os.path.splitext(os.path.basename(x))[0] + '.png')
    df['cropped_image_zoomout4_1024_path'] = df[0].map(lambda x: os.path.split(os.path.split(x)[0])[0] + '/images_detectron2box_zoomout4_1024/' + os.path.splitext(os.path.basename(x))[0] + '.png')

    df = df[~df.cropped_image_zoomout4_1024_path.map(lambda x: os.path.isfile(x))]
    df.reset_index(inplace=True, drop=True)

    zoom_factors = [2, 2.5, 3, 4]

    for idx in tqdm(df.index.tolist()):
        img_loc = df.loc[idx, 0]
        box_cx, box_cy, box_w, box_h, angle = df.loc[idx, 1]

        if box_w < 1:
            box_w = 1
        if box_h < 1:
            box_h = 1

        img = ImageOps.exif_transpose(Image.open(img_loc)).convert('RGB')

        crop_and_save_images(img, box_cx, box_cy, box_w, box_h, df, idx, zoom_factors)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Crop and save images.")
    # here input files should be in first column and bounding box in second column
    parser.add_argument("input_file", type=str, help="Path to the input dataframe file")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
