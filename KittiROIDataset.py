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
        self.num = 0
        self.img_files = []
        for file in os.listdir(self.dir):
            if fnmatch.fnmatch(file, '*.png'):
                self.img_files += [file]
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
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        label = int(self.labels[idx])
        return image, label

    def __iter__(self):
        self.num = 0
        return self

    def __next__(self):
        if (self.num >= self.max):
            raise StopIteration
        else:
            self.num += 1
            return self.__getitem__(self.num - 1)

    class_label = {'NoCar': 0, 'Car': 1}
