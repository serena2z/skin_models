import numpy as np
import os
import torch
from utils import get_labelme_dataset_function
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, detection_utils as utils, transforms as T, build_detection_train_loader
from detectron2.engine import launch, DefaultTrainer
from detectron2.structures import BoxMode
from detectron2.evaluation import RotatedCOCOEvaluator, DatasetEvaluators
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
import argparse

def rotate_bbox(annotation, transforms):
    annotation["bbox"] = transforms.apply_rotated_box(
        np.asarray([annotation['bbox']]))[0]
    annotation["bbox_mode"] = BoxMode.XYXY_ABS
    return annotation

def get_shape_augmentations():
    # Optional shape augmentations
    return [
        T.RandomFlip(),
        T.ResizeShortestEdge(short_edge_length=(
            640, 672, 704, 736, 768, 800), max_size=1333, sample_style='choice'),
        T.RandomFlip()
    ]

def get_color_augmentations():
    # Optional color augmentations
    return T.AugmentationList([
        T.RandomBrightness(0.9, 1.1),
        T.RandomSaturation(intensity_min=0.75, intensity_max=1.25),
        T.RandomContrast(intensity_min=0.76, intensity_max=1.25)
    ])

def dataset_mapper(dataset_dict):
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    color_aug_input = T.AugInput(image)
    get_color_augmentations()(color_aug_input)
    image = color_aug_input.image
    image, image_transforms = T.apply_transform_gens(
        get_shape_augmentations(), image)
    dataset_dict["image"] = torch.as_tensor(
        image.transpose(2, 0, 1).astype("float32"))

    annotations = [
        rotate_bbox(obj, image_transforms)
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]
    instances = utils.annotations_to_instances_rotated(
        annotations, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    return dataset_dict

class RotatedBoundingBoxTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluators = [RotatedCOCOEvaluator(
            dataset_name, cfg, True, output_folder)]
        return DatasetEvaluators(evaluators)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=dataset_mapper)

def train_detectron(train_directory, test_directory, save_dir, device):
    class_labels = ["lesion"]

    train_dataset_function = get_labelme_dataset_function(
        train_directory, class_labels)
    train_dataset_name = "train_ship_dataset"
    DatasetCatalog.register(train_dataset_name, train_dataset_function)
    MetadataCatalog.get(train_dataset_name).set(thing_classes=["lesion"])

    test_dataset_function = get_labelme_dataset_function(
        test_directory, class_labels)
    test_dataset_name = "test_ship_dataset"
    DatasetCatalog.register(test_dataset_name, test_dataset_function)
    MetadataCatalog.get(test_dataset_name).set(thing_classes=["lesion"])

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.merge_from_file(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "rotated_bbox_config_mod.yaml"))
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (test_dataset_name,)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    cfg.OUTPUT_DIR = save_dir
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_labels)
    cfg.MODEL.DEVICE='cpu'

    trainer = RotatedBoundingBoxTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

def main():
    parser = argparse.ArgumentParser(description="Train a model using Detectron2 with rotated bounding boxes.")
    parser.add_argument("--train_dir", type=str, help="Path to the training dataset directory")
    parser.add_argument("--test_dir", type=str, help="Path to the testing dataset directory")
    parser.add_argument("--save_dir", type=str, default="./model", help="Directory to save the model")
    parser.add_argument("--device", type=str, default="cpu", help="Device to train the model on")
    args = parser.parse_args()

    train_detectron(args.train_dir, args.test_dir, args.save_dir, args.device)

if __name__ == "__main__":
    main()
