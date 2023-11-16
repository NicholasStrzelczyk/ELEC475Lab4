from PIL import Image
from torch.utils.data import Dataset


class ROIBatchDataset(Dataset):
    def __init__(self, data, transform=None):
        super().__init__()
        self.transform = transform
        self.images = [Image.fromarray(image) for image, _ in data]
        self.labels = [label for _, label in data]
        # create a parameter to keep track of each ROI's box coords from orig image

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        image_sample = self.transform(image)
        label = self.labels[index]
        return image_sample, label
