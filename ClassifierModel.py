from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class ClassifierModel(nn.Module):

    def __init__(self):
        super(ClassifierModel, self).__init__()
        self.resnet18 = resnet18(weights=ResNet18_Weights.DEFAULT)
        # self.resnet18 = resnet18(weights=None)
        in_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(in_features=in_features, out_features=2)

    def forward(self, x):
        return self.resnet18(x)
