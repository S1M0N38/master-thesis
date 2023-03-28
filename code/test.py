import argparse
import pathlib

import advertorch
import data
import models
import torch
import torchvision.transforms as transforms
from tester import Tester

print(
    """
Table of Experiments:
+--------------------------------------------+
| --name                                     |
|============================================|
| Che+19a_ResNet110_CIFAR10_XE_0200.pth      |
| Che+19a_ResNet110_CIFAR100_XE_0200.pth     |
| Che+19b_ResNet56_CIFAR10_GCE_0200.pth      |
| Che+19b_ResNet56_CIFAR10_XE_0200.pth       |
| Che+19b_ResNet56_CIFAR100_GCE_0200.pth     |
| Che+19b_ResNet56_CIFAR100_XE_0200.pth      |
| Che+19c_ResNet56_CIFAR100_HGCE_0200.pth    |
| Che+19c_ResNet56_CIFAR100_XE,HCE_0200.pth  |
+--------------------------------------------+
"""
)

PATH_HERE = pathlib.Path(__file__).parent
PATH_MODELS = PATH_HERE / "models"
PATH_RUNS = PATH_HERE / "runs"

# CLI ##################################################################################

parser = argparse.ArgumentParser(
    description="Choose which Model to test against attack.",
    epilog="Source code: https://github.com/S1M0N38/master-thesis",
)
parser.add_argument(
    "-n",
    "--name",
    help="Select a model to test.",
    required=True,
)
parser.add_argument(
    "-a",
    "--attack",
    action="store_true",
    help="Perfrom various adversarial attacks.",
)
args = parser.parse_args()

# PARSE NAME ###########################################################################

tag, model, dataset, loss, epochs = args.name.split("_")
device: str = "cuda" if torch.cuda.is_available() else "cpu"

if dataset == "CIFAR10":
    testloader = data.testloader_CIFAR10
    num_classes = 10
    lca = data.LCA_CIFAR10
elif dataset == "CIFAR100":
    testloader = data.testloader_CIFAR100
    num_classes = 100
    lca = data.LCA_CIFAR100
else:
    raise NotImplementedError

if model == "ResNet56":
    model = models.ResNet56(num_classes=num_classes)
elif model == "ResNet110":
    model = models.ResNet110(num_classes=num_classes)
else:
    raise NotImplementedError
checkpoint = torch.load(PATH_MODELS / args.name, map_location=device)
model.load_state_dict(checkpoint["model"])

if args.attack:
    attacks = {
        "FGSM": advertorch.attacks.FGSM(model, loss_fn=torch.nn.CrossEntropyLoss()),
        "I-FGSM": advertorch.attacks.LinfBasicIterativeAttack(model),
        "PGD": advertorch.attacks.PGDAttack(model),
    }
else:
    attacks = {}

pre_transform = transforms.Normalize(
    mean=(-0.4914 / 0.2023, -0.4822 / 0.1994, -0.4465 / 0.2010),
    std=(1 / 0.2023, 1 / 0.1994, 1 / 0.2010),
)
post_transform = transforms.Normalize(
    mean=(0.4914, 0.4822, 0.4465),
    std=(0.2023, 0.1994, 0.2010),
)

# ATTACK ###############################################################################

tester = Tester(
    model=model,
    testloader=testloader,
    lca=lca,
    attacks=attacks,
    pre_transform=pre_transform,
    post_transform=post_transform,
    device=device,
)

results = tester.test()

# PRINT RESULTS #######################################################################
s = f"{args.name}\n"
for attack, metrics in results.items():
    for metric_name, metric_value in metrics.items():
        s += f"| {attack} {metric_name}: {metric_value.item():.4f} "
print(s)
