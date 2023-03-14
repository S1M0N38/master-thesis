import pathlib

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

here = pathlib.Path(__file__).parent
datasets_path = here / "datasets"

BATCH_SIZE = 4
DOWNLOAD = True

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

trainset = torchvision.datasets.CIFAR10(
    str(datasets_path),
    train=True,
    transform=transform,
    download=DOWNLOAD,
)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE)

testset = torchvision.datasets.CIFAR10(
    str(datasets_path),
    train=False,
    transform=transform,
    download=DOWNLOAD,
)
testloader = DataLoader(testset, batch_size=BATCH_SIZE)

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
