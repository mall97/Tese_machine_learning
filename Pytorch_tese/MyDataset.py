import os 
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io
import torchvision.transforms as transforms


class MyDataset(Dataset):
    def __init__(self, csv_file, dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.dir = dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = os.path.join(self.dir, self.data.iloc[index, 0])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.data.iloc[index, 1]))
        if self.transform:
            image=self.transform(image)
        return (image, y_label)