import pandas as pd
from PIL import Image, ImageOps
import os
import json
from tqdm import tqdm
import argparse

def create_detectron_json(df_file, output_dir):
    # read in the dataframe file
    df = pd.read_csv(df_file, index_col=0)
    df.reset_index(drop=True, inplace=True)

    # detectron index
    df.loc[df[df.subset == 'train'].index, 'detectron_index'] = range(len(df[df.subset == 'train']))
    df.loc[df[df.subset == 'test'].index, 'detectron_index'] = range(len(df[df.subset == 'test']))
    df.loc[df[df.subset == 'val'].index, 'detectron_index'] = range(len(df[df.subset == 'val']))
    df['detectron_index'] = df['detectron_index'].astype('int')

    os.makedirs(output_dir, exist_ok=True)

    # making dictionary
    for i in tqdm(range(len(df))):
        img_location = df.file_name.iloc[i]
        img_name = os.path.basename(img_location)

        img = ImageOps.exif_transpose(Image.open(img_location).convert("RGB"))
        width, height = img.size

        x1, y1, x2, y2, x3, y3, x4, y4 = df[['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']].iloc[i].tolist()

        detectron_idx = df.detectron_index.iloc[i]
        subset = df.subset.iloc[i]

        Dict = {"version": "5.0.1", "flags": {}}

        Dict["orig_name"] = img_name
        Dict["orig_path"] = img_location
        Dict["shapes"] = [{
            "label": "lesion",
            "points": (
                (x1, y1),
                (x2, y2),
                (x3, y3),
                (x4, y4)
            ),
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        }]

        Dict["imagePath"] = str(detectron_idx) + '.png'
        Dict["imageData"] = None
        Dict["imageHeight"] = height
        Dict["imageWidth"] = width

        # saving everything
        dir_save = os.path.join(output_dir, subset)
        os.makedirs(dir_save, exist_ok=True)
        dict_save = os.path.join(dir_save, str(detectron_idx) + '.json')
        img_save = os.path.join(dir_save, str(detectron_idx) + '.png')

        img.save(img_save)

        # Serializing json
        json_object = json.dumps(Dict, indent=4)
        # Writing to sample.json
        with open(dict_save, "w") as outfile:
            outfile.write(json_object)

def main():
    parser = argparse.ArgumentParser(description="Convert annotations to Detectron2 JSON format.")
    parser.add_argument("df_file", type=str, help="Path to the dataframe file")
    parser.add_argument("output_dir", type=str, help="Directory to save the output JSON files")
    args = parser.parse_args()

    create_detectron_json(args.df_file, args.output_dir)

if __name__ == "__main__":
    main()
