import cv2
import os
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import numpy as np
import argparse

def inference(image_paths, weights, output_file, device):
    class_labels = ["lesion"]

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.merge_from_file(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "rotated_bbox_config_mod.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_labels)
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0
    cfg.MODEL.DEVICE = device

    predictor = DefaultPredictor(cfg)

    # make it so that it overwrites the file if it already exists
    with open(output_file, 'w') as fh:
        if os.path.getsize(fh.name) == 0:
            fh.write("image_path,highest_score_box,box_points,box_points_square,highest_score\n")
        for image_path in image_paths:
            img = cv2.imread(image_path)
            outputs = predictor(img)

            # Save the inference results to a text file
            save_results(fh, image_path, outputs)

def save_results(fh, image_path, outputs):
    # file paths
    fh.write(image_path)
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
        box_points = np.intp(box_points)  # Convert to integer

        # Save highest_score_box
        # safe highest_score_box [[center_x,center_y],[w,h],angle]
        fh.write('"[')
        for ii, line in enumerate(highest_score_box):
            if ii != 0:
                fh.write(f",{line}")
            else:
                fh.write(f"{line}")
        fh.write(']"')

        # save box_points
        box_points_str = ','.join([f"[{x},{y}]" for x, y in box_points])
        fh.write(',"[')
        fh.write(box_points_str)
        fh.write(']"')

        # Save box_points_square
        wh_square = [np.average(highest_score_box[2:4]), np.average(highest_score_box[2:4])]
        box_points_square = cv2.boxPoints((highest_score_box[:2], wh_square, highest_score_box[4]))
        box_points_square = np.intp(box_points_square)

        # Format box_points_square as a string with commas separating coordinates
        box_points_square_str = ','.join([f"[{x},{y}]" for x, y in box_points_square])

        fh.write(',"[')
        fh.write(box_points_square_str)  # Write formatted box_points_square
        fh.write(']"')

        fh.write(',' + str(highest_score) + '\n')

def main():
    # change to folder
    parser = argparse.ArgumentParser(description="Perform inference using a trained model.")
    parser.add_argument("--img_folder", type=str, help="Path to the image folder", required=True)
    parser.add_argument("--model", type=str, help="Path to the trained model weights file", required=True)
    parser.add_argument("--output_file", type=str, default="./predicted.txt", help="Path to the output file for saving inference results")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for inference (default: cpu)")
    args = parser.parse_args()

    img_folder = args.img_folder
    model = args.model
    output_file = args.output_file
    device = args.device

    image_paths = [f for f in os.listdir(img_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    image_paths = [os.path.join(img_folder, f) for f in image_paths]
    inference(image_paths, model, output_file, device)

if __name__ == "__main__":
    main()
