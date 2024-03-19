import cv2
import os
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
import numpy as np
import pandas as pd
from tqdm.contrib import tzip
import argparse

def inference(image_paths, weights, out_paths):
    class_labels = ["lesion"]
    metadata = MetadataCatalog.get("ship_dataset").set(thing_classes=class_labels)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.merge_from_file("rotated_bbox_config_mod.yaml")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_labels)
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0

    predictor = DefaultPredictor(cfg)

    for image_path, out_path in tzip(image_paths, out_paths):
        img = cv2.imread(image_path)
        outputs = predictor(img)

        # Save the inference results to a text file
        save_results(image_path, out_path, outputs)

def save_results(image_path, out_path, outputs):
    with open('predicted_boxes.txt','a+') as fh:
        if os.path.getsize('predicted_boxes.txt') == 0:
            fh.write("image_path,out_path,box_points,box_points_square,highest_score\n")

        # file paths
        fh.write(image_path)
        fh.write(',')
        fh.write(out_path)
        fh.write(',')
        
        pred_boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
        scores = outputs["instances"].scores.cpu().numpy()

        if len(scores) > 0:
            # Find the index of the box with the highest score
            highest_score_index = scores.argmax()

            # Select the box and score with the highest confidence
            highest_score_box = pred_boxes[highest_score_index]
            highest_score = scores[highest_score_index]

            # Convert the rotated box to a set of points
            box_points = cv2.boxPoints((highest_score_box[:2], highest_score_box[2:4], highest_score_box[4]))
            box_points = np.int0(box_points)  # Convert to integer

            # Save highest_score_box
            fh.write('"[')
            for ii, line in enumerate(highest_score_box):
                if ii != 0:
                    fh.write(f",{line}")
                else:
                    fh.write(f"{line}")
            fh.write(']"')

            fh.write(',' + str(highest_score) + '\n')

            # Save box_points_square
            wh_square = [np.average(highest_score_box[2:4]), np.average(highest_score_box[2:4])]
            box_points_square = cv2.boxPoints((highest_score_box[:2], wh_square, highest_score_box[4]))
            box_points_square = np.int0(box_points_square)

            fh.write(',"[')
            for ii, line in enumerate(box_points_square):
                if ii != 0:
                    fh.write(f",{line}")
                else:
                    fh.write(f"{line}")
            fh.write(']"')

            fh.write(',' + str(highest_score) + '\n')

def main(args):
    df = pd.read_csv(args.csv_file, index_col=0)
    image_paths = df.image_path.tolist()
    out_paths = df.predmask_boxoverlay_path.tolist()
    inference(image_paths, args.weights, out_paths)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform inference using a trained model.")
    parser.add_argument("--csv_file", type=str, help="Path to the CSV file containing image paths", required=True)
    parser.add_argument("--weights", type=str, help="Path to the trained model weights file", required=True)
    args = parser.parse_args()
    main(args)
