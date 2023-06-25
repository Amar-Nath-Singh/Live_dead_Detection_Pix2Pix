from PIL import Image

import numpy as np
import os
from torch.utils.data import Dataset

from config import *
import cv2

class CellDataset(Dataset):
    def __init__(self, list_img) -> None:
        super(CellDataset).__init__()
        self.list_files = list_img

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        alive_path = img_file[:-5]+"1.jpg"
        dead_path = img_file[:-5]+"2.jpg"
        input_image = np.array(Image.open(img_file))
        alive = np.array(Image.open(alive_path))
        dead = np.array(Image.open(dead_path))

        # input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        target_image = cv2.add(alive, dead)
        # gray_cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        # ret, mask = cv2.threshold(gray_cell, 25, 255, cv2.THRESH_BINARY)
        # inv_mask = cv2.bitwise_not(mask)
        # target_image = cv2.add(cell, cv2.bitwise_and(input_image, input_image, mask=inv_mask))

        augmentation = both_transform(image=input_image, image0=target_image)

        input_image, target_image = augmentation["image"], augmentation["image0"]
        input_image = transform_only_input(image = input_image)["image"]
        target_image = transform_only_mask(image = target_image)["image"]

        return input_image, target_image