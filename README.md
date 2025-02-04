# Skin Models

Link to paper:

# Table of Contents

- [Installation](#installation)
- [Lesion Segmenter](#lesion-segmenter)
- [Distance Classifier](#distance-classifier)
- [Skin and Ruler Segmenters](#skin-and-ruler-segmenters)
- [Purple Segmenter](#purple-segmenter)

## Installation

To use the skin_model package in your local Python environment, you'll need to follow these steps:

### Clone the Repository

Clone the project repository from GitHub using the following command:

```
git clone https://github.com/serena2z/skin_models.git
```

### Create and Activate Conda Environment

Navigate to the cloned repository directory and create a conda environment using the provided environment.yml file:

```
cd skin_models
conda env create -f environment.yml
```
Activate the newly created conda environment:

```
conda activate skin_model
```

### Run the App

Once the environment is set up and activated, you can run the skin models according to the instructions provided in the respective Python scripts or readme.

## Lesion Segmenter

### Inference

To perform inference using our trained model, follow these steps:

1. **Navigate to the Inference Folder**

  Go to the `inference` folder in the project repository. Here's what you'll find:
  
  - `inference.py`: Main script for running the model and computing the cropped box coordinates around each lesion.
  - `box_overlay.py`: Handles visualizations of the cropped box around the lesion.
  - `crop_img.py`: Generates a crop of the image to the cropped box.
  - `samples`: Contains sample files that you can use.

2. **Run Inference**

  To run `inference.py`, you'll need a CSV file with all your image names (the complete path) in one column. Use the following command in the terminal:
  
  ```
  python inference.py --img_folder IMG_FOLDER --model MODEL [--output_file OUTPUT_FILE] [--device DEVICE]
  ```
  ```
  - `IMG_FOLDER`: Path to the folder where your images are stored.
  - `MODEL`: Path to the trained model weights file.
  - `OUTPUT_FILE` [OPTIONAL]: Path to the output file for saving inference results, default is box_predictions.txt.
  - `DEVICE` [OPTIONAL]: Device to use for inference, default is cpu.
  ```

  The output generated from `inference.py` will be a text file with the images and the segmented box coordinates. In particular, you'll see:
  
  - `image_path`: Column for image path
  - `center_x`: x-coordinate of the center of the box
  - `center_y`: y-coordinate of the center of box
  - `width`: width of the box
  - `height`: height of the box
  - `angle`: angle of rotation of the box
  - `box_points`: All 4 box corner coordinates in the form of [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
  - `box_points_square`: Same as `box_points` but resizing the corner coordinates to form a square box instead of a rectangular box.
  - `highest_score`: Score of the best fitting box.

3. **Visualize Boxes**

  To visualize these boxes on the original images, use `box_overlay.py`.
  
  ```
  box_overlay.py --input_file INPUT_FILE --save_dir SAVE_DIR [--thickness THICKNESS] [--resize]
  ```
  ```
  - `INPUT_FILE`: Path to the text file containing results from inference.py, default is box_predictions.txt.
  - `SAVE_DIR`: Name of the output folder to save the image visualizations.
  - `THICKNESS` [OPTIONAL]: Thickness used to visualize the boundaries of the boxes, default is 2.
  - `RESIZE` [OPTIONAL]: If you include this flag, it will resize the smaller side of the image to 512 while keeping aspect ratio constant to reduce image storage, default is false.
  ```

4. **Crop Images**

  Lastly, you can crop your images to the lesion using `crop_image.py`.
  
  ```
  crop_img.py --input_file INPUT_FILE --save_dir SAVE_DIR
  ```
  ```
  - `INPUT_FILE`: Path to the text file containing results from inference.py, default is box_predictions.txt.
  - `SAVE_DIR`: Name of the output folder to save the cropped image.
  ```

  These organized steps streamline the process of performing inference and post-processing on lesion images using our model.

### Train

To train your own model in the same format,

## Distance Classifier

The `distance_classification` folder has the following files:
- `run.py`: Main script for running the classification model.
- `train.py`: Training the distance classifier.
- `metrics.py`: Compute metrics if the true distances are known.

To run inference on your images, you can use the following command:

```
  python run.py --input_file INPUT_FILE --model_path MODEL_PATH [--save_path SAVE_PATH] [--device DEVICE] [--batch_size BATCH_SIZE] [--num_workers NUM_WORKERS]
```

```
- `INPUT_FILE`: Path to the predictions from the lesion cropper or an input file that has image_path, center_x and center_y (coordinates of the center of the lesion box)
- `MODEL`: Path to the trained model weights file.
- `OUTPUT_FILE` [OPTIONAL]: Path to the output file for saving inference results, default is distance_predictions.csv.
- `DEVICE` [OPTIONAL]: Device to use for inference, default is cpu.
- `BATCH_SIZE` [OPTIONAL]: Batch size to use for inference, default is 2.
- `NUM_WORKERS` [OPTIONAL]: Number of workers to use for inference, default is 1.
```

The output generated from `inference.py` will be a text file with the images and the predicted distances:

- `image_path`: Column for image path
- `pred_pxcm`: Predicted pixels per centimeter
- `pred_pxcm224`: Predicted pixels per centimeter resized.


## Skin and Ruler Segmenters

Both `skin_segmenter` and `ruler_segmenter` folders will have the following important files:
- `run.py`: Main script for running the segmentation models.
- `test.py`: Evaluating the skin/ruler segmenter.
- `train.py`: Training the skin/ruler segmenter.
- `samples`: Contains sample files that you can use.

## Purple Segmenter

