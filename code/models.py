import torch.nn as nn
import torch.nn.functional as F
import torchvision


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        f0 = x = self.flatten(x)
        f1 = x = F.relu(self.fc1(x))
        f2 = x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x, (f0, f1, f2)


class ResNet10(nn.Module):
    def __init__(self):
        super().__init__()
        weights = torchvision.models.ResNet50_Weights.DEFAULT
        self.preprocess = weights.transforms()
        self.resnet = torchvision.models.resnet50(weights=weights)

        # Freeze params for resnet
        # for params in self.resnet.parameters():
        #     params.requires_grad = False

        # Feature extraction for CIFAR-10
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 10)

    def forward(self, x):
        x = self.preprocess(x)
        x = self.resnet(x)
        return x, None
