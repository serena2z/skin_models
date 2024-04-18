import cv2
import os
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import numpy as np
import argparse
import csv
from tqdm import tqdm

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

    with open(output_file, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(["image_path", "center_x", "center_y", "width","height", "angle", "box_points", "box_points_square", "highest_score"])
        for idx, image_path in enumerate(tqdm(image_paths)):
            img = cv2.imread(image_path)
            outputs = predictor(img)
            save_results(writer, image_path, outputs)


def save_results(writer, image_path, outputs):
    pred_boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
    scores = outputs["instances"].scores.cpu().numpy()

    if len(scores) > 0:
        highest_score_index = scores.argmax()
        # highest_score_box [center_x,center_y,w,h,angle]
        highest_score_box = pred_boxes[highest_score_index]
        highest_score = scores[highest_score_index]

        box_points = cv2.boxPoints((highest_score_box[:2], highest_score_box[2:4], highest_score_box[4]))
        box_points = np.intp(box_points).tolist()

        wh_square = [np.average(highest_score_box[2:4]), np.average(highest_score_box[2:4])]
        box_points_square = cv2.boxPoints((highest_score_box[:2], wh_square, highest_score_box[4]))
        box_points_square = np.intp(box_points_square).tolist()

        writer.writerow([image_path,
                            highest_score_box[0],
                            highest_score_box[1],
                            highest_score_box[2],
                            highest_score_box[3],
                            highest_score_box[4],
                            box_points,
                            box_points_square,
                            highest_score])
            

def main():
    # change to folder
    parser = argparse.ArgumentParser(description="Perform inference using a trained model.")
    parser.add_argument("--img_folder", type=str, help="Path to the image folder", required=True)
    parser.add_argument("--model", type=str, help="Path to the trained model weights file", required=True)
    parser.add_argument("--output_file", type=str, default="box_predictions.csv", help="Path to the output file for saving inference results")
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
