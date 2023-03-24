import pathlib
import random

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

PATH_HERE = pathlib.Path(__file__).parent
PATH_DATASETS = PATH_HERE / "datasets"

DOWNLOAD = False
NUM_WORKERS = 4

# TODO: fix error cause by
#   num_workers=NUM_WORKERS,
#   worker_init_fn=seed_worker,
#   generator=g,
# when running attack. Temp fix is to comment those lines.


# Preserve reproducibility
# https://pytorch.org/docs/stable/notes/randomness.html#dataloader
def seed_worker(_):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)

# CIFAR-10 #############################################################################

batch_size_CIFAR10 = 128

classes_CIFAR10 = (
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

# Trasnformations of train and test datasets follow ChenMar19a
transform_train_CIFAR10 = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)
transform_test_CIFAR10 = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

# Using CIFAR10 builtin datasets
trainset_CIFAR10 = torchvision.datasets.CIFAR10(
    str(PATH_DATASETS),
    train=True,
    transform=transform_train_CIFAR10,
    download=DOWNLOAD,
)
testset_CIFAR10 = torchvision.datasets.CIFAR10(
    str(PATH_DATASETS),
    train=False,
    transform=transform_test_CIFAR10,
    download=DOWNLOAD,
)

# Convert to iterable with batch_size=batch_size_CIFAR10
trainloader_CIFAR10 = DataLoader(
    trainset_CIFAR10,
    batch_size=batch_size_CIFAR10,
    num_workers=NUM_WORKERS,
    worker_init_fn=seed_worker,
    generator=g,
)
testloader_CIFAR10 = DataLoader(
    testset_CIFAR10,
    batch_size=batch_size_CIFAR10,
    num_workers=NUM_WORKERS,
    worker_init_fn=seed_worker,
    generator=g,
)

# CIFAR-100 ############################################################################

batch_size_CIFAR100 = 128

classes_CIFAR100 = (
    ("beaver", "dolphin", "otter", "seal", "whale"),
    ("aquarium_fish", "flatfish", "ray", "shark", "trout"),
    ("orchid", "poppy", "rose", "sunflower", "tulip"),
    ("bottle", "bowl", "can", "cup", "plate"),
    ("apple", "mushroom", "orange", "pear", "sweet_pepper"),
    ("clock", "keyboard", "lamp", "telephone", "television"),
    ("bed", "chair", "couch", "table", "wardrobe"),
    ("bee", "beetle", "butterfly", "caterpillar", "cockroach"),
    ("bear", "leopard", "lion", "tiger", "wolf"),
    ("bridge", "castle", "house", "road", "skyscraper"),
    ("cloud", "forest", "mountain", "plain", "sea"),
    ("camel", "cattle", "chimpanzee", "elephant", "kangaroo"),
    ("fox", "porcupine", "possum", "raccoon", "skunk"),
    ("crab", "lobster", "snail", "spider", "worm"),
    ("baby", "boy", "girl", "man", "woman"),
    ("crocodile", "dinosaur", "lizard", "snake", "turtle"),
    ("hamster", "mouse", "rabbit", "shrew", "squirrel"),
    ("maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"),
    ("bicycle", "bus", "motorcycle", "pickup_truck", "train"),
    ("lawn_mower", "rocket", "streetcar", "tank", "tractor"),
)

# fmt: off
fine_to_coarse_CIFAR100 = [
     4,  1, 14,  8,  0,  6,  7,  7, 18,  3,  3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
     6, 11,  5, 10,  7,  6, 13, 15,  3, 15,  0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
     5, 19,  8,  8, 15, 13, 14, 17, 18, 10, 16,  4, 17,  4,  2,  0, 17,  4, 18, 17,
    10,  3,  2, 12, 12, 16, 12,  1,  9, 19,  2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
    16, 19,  2,  4,  6, 19,  5,  5,  8, 19, 18,  1,  2, 15,  6,  0, 17,  8, 14, 13,
]
# fmt: on

# Keep using Che+19a normalization
transform_train_CIFAR100 = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)
transform_test_CIFAR100 = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

# Using CIFAR100 builtin datasets
trainset_CIFAR100 = torchvision.datasets.CIFAR100(
    str(PATH_DATASETS),
    train=True,
    transform=transform_train_CIFAR100,
    download=DOWNLOAD,
)
testset_CIFAR100 = torchvision.datasets.CIFAR100(
    str(PATH_DATASETS),
    train=False,
    transform=transform_test_CIFAR100,
    download=DOWNLOAD,
)

# Convert to iterable with batch_size=batch_size_CIFAR100
trainloader_CIFAR100 = DataLoader(
    trainset_CIFAR100,
    batch_size=batch_size_CIFAR100,
    num_workers=NUM_WORKERS,
    worker_init_fn=seed_worker,
    generator=g,
)
testloader_CIFAR100 = DataLoader(
    testset_CIFAR100,
    batch_size=batch_size_CIFAR100,
    num_workers=NUM_WORKERS,
    worker_init_fn=seed_worker,
    generator=g,
)

if __name__ == "__main__":
    breakpoint()
