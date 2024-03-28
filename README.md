# Skin Models

Link to paper:

# Table of Contents

- [Installation](#installation)
- [Lesion Segmenter](#lesion-segmenter)
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
  - `crop_img.py`: Allows you to crop the image to the cropped box.
  - `samples`: Contains sample files that you can use.

2. **Run Inference**

  To run `inference.py`, you'll need a CSV file with all your image names (the complete path) in one column. Use the following command in the terminal:
  
  ```
  python inference.py --csv_file CSV_FILE --col_name COL_NAME --model MODEL --output_file OUTPUT_FILE [--device DEVICE]
  ```
  
  The output generated from `inference.py` will be a text file with the images and the segmented box coordinates. In particular, you'll see:
  
  - `image_path`: Column for image path
  - `highest_score_box`:  [[center_x,center_y],[w,h],angle]
  - `box_points`: All 4 box coordinates in the form of [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
  - `box_points_square`: Same as `box_points` but using a square box instead of a rectangular box.
  - `highest_score`: Score of the best fitting box.

3. **Visualize Boxes**

  To visualize these boxes on the original images, use `box_overlay.py` with your text file output from `inference.py` as input.
  
  ```
  box_overlay.py [-h] --txt_file TXT_FILE --output_dir OUTPUT_DIR
  ```

4. **Crop Images**

  Lastly, you can crop your images to the lesion using `crop_image.py` with the same text file output from `inference.py` as input.
  
  ```
  crop_images.py [-h] --input_file INPUT_FILE --save_dir SAVE_DIR
  ```
  
  These organized steps streamline the process of performing inference and post-processing on lesion images using our model.

### Train

To train your own model in the same format,

## Skin and Ruler Segmenters

Both `skin_segmenter` and `ruler_segmenter` folders will have the following important files:
- `run.py`: Main script for running the segmentation models.
- `test.py`: Evaluating the skin/ruler segmenter.
- `train.py`: Training the skin/ruler segmenter.
- `samples`: Contains sample files that you can use.

## Purple Segmenter

