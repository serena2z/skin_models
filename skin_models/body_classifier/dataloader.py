import numpy as np
import torchvision.transforms as tf
from torch.utils.data import Dataset
from PIL import Image

WIDTH = HEIGHT = 224

class RunImageDataset(Dataset):
    def __init__(self, df, img_path):
        super().__init__()
        self.df = df
        self.img_path = img_path
        self.label_dict = {"face": 0, "scalp": 1, "neck": 2, "arms": 3, "hands": 4, "chest and abdomen": 5, 
                           "back": 6, "legs": 7, "genital and perianal": 8, "feet": 9, "dermoscope": 10}

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        img_location = self.df[self.img_path].iloc[index]

        try:
            image = Image.open(img_location).convert('RGB')

        except:
            # choose another random image to load instead
            random_idx = np.random.choice(self.df.shape[0])
            with open('none_importable_images.txt','a+') as fh:
                fh.write(img_location +',' + self.df[self.img_path].iloc[random_idx] + '\n')
            return self.__getitem__(random_idx)

        image = self.transform(image)
        return image, img_location
    
    def transform(self, image: Image.Image) -> Image.Image:
        transform_ops = tf.Compose([
            tf.Resize((HEIGHT, WIDTH)),
            tf.ToTensor(),
            tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return transform_ops(image)