from PIL import Image
from torch.utils.data import Dataset


class ROIBatchDataset(Dataset):
    def __init__(self, data, transform=None):
        super().__init__()
        self.transform = transform
        self.images = [Image.fromarray(image) for image, _, _ in data]
        self.labels = [label for _, label, _ in data]
        self.bboxes = [bbox for _, _, bbox in data]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        image_sample = self.transform(image)
        label = self.labels[index]
        bbox = self.bboxes[index]
        return image_sample, label, bbox
