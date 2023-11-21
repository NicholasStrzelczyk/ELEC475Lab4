import os
import fnmatch
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms, models
import cv2


class KittiROIDataset(Dataset):
    def __init__(self, dir, training=True, transform=None):
        self.training = training
        self.mode = 'train'
        if self.training == False:
            self.mode = 'test'
        self.dir = os.path.join(dir, self.mode)
        self.transform = transform

        unordered_files = os.listdir(self.dir)
        ordered_files = sorted(unordered_files, key=self.sort_my_ROIs)
        self.img_files = []
        for file in ordered_files:
            if fnmatch.fnmatch(file, '*.png'):
                self.img_files.append(file)

        self.labels = []
        label_path = os.path.join(self.dir, 'labels.txt')
        with open(label_path) as label_file:
            labels_string = label_file.readlines()
        for i in range(len(labels_string)):
            lsplit = labels_string[i].split(' ')
            label = lsplit[1]
            self.labels.append(label)
        self.max = len(self)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.dir, self.img_files[idx])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = torch.from_numpy(image).float()
        image = image.permute(2, 0, 1)
        image = self.transform(image)
        label = int(self.labels[idx])
        return image, label

    class_label = {'NoCar': 0, 'Car': 1}

    def sort_my_ROIs(self, file):
        sort_tuple = (1, 0, 0)
        filename = file.split(".")
        if filename[1] == "png":
            img_num, roi_num = filename[0].split("_")
            sort_tuple = (0, int(img_num), int(roi_num))
        return sort_tuple
