import argparse

import data
import losses
import models
import torch
from trainer import Trainer

torch.manual_seed(0)

print(
    """
Table of Experiments:
+-----------------------------------------------------+
| --tag   | --model   | --dataset | --loss | --epochs |
|=====================================================|
| Che+19a | ResNet110 | CIFAR10   | XE     | 200      |
| Che+19a | ResNet110 | CIFAR10   | XE,CE  | 200      |
| Che+19a | ResNet110 | CIFAR100  | XE     | 200      |
| Che+19a | ResNet110 | CIFAR100  | XE,CE  | 200      |
|---------+-----------+-----------+--------+----------|
| Che+19b | ResNet56  | CIFAR10   | XE     | 200      |
| Che+19b | ResNet56  | CIFAR10   | GCE    | 200      |
| Che+19b | ResNet56  | CIFAR100  | XE     | 200      |
| Che+19b | ResNet56  | CIFAR100  | GCE    | 200      |
|---------+-----------+-----------+--------+----------|
| Che+19c | ResNet56  | CIFAR100  | XE,CE  | 200      |
| Che+19c | ResNet56  | CIFAR100  | XE,HCE | 200      |
| Che+19c | ResNet56  | CIFAR100  | HGCE   | 200      |
+-----------------------------------------------------+
Note: The optimizer[s] and lr_scheduler[s] are the same for all experiments

Training process produce:
  - models/[TAG_]{MODEL}_{DATASET}_{LOSS}_{EPOCHS}.pth
  - runs/[TAG_]{MODEL}_{DATASET}_{LOSS}/*
"""
)


# CLI ##################################################################################

parser = argparse.ArgumentParser(
    description="Train pytorch models.",
    epilog="Source code: https://github.com/S1M0N38/master-thesis",
)
parser.add_argument(
    "-t",
    "--tag",
    type=str,
    action="store",
    default="",
)
parser.add_argument(
    "-m",
    "--model",
    type=str,
    action="store",
    choices=["ResNet56", "ResNet110"],
    required=True,
)
parser.add_argument(
    "-d",
    "--dataset",
    type=str,
    action="store",
    choices=["CIFAR10", "CIFAR100"],
    required=True,
)
parser.add_argument(
    "-l",
    "--loss",
    type=str,
    action="store",
    choices=["XE", "XE,CE", "GCE", "XE,HCE", "HGCE"],
    required=True,
)
parser.add_argument(
    "-e",
    "--epochs",
    type=int,
    required=True,
)
# TODO: Add gpu selection

args = parser.parse_args()


# PROCESS FLAGS ########################################################################

if args.dataset == "CIFAR10":
    trainloader = data.trainloader_CIFAR10
    testloader = data.testloader_CIFAR10
elif args.dataset == "CIFAR100":
    trainloader = data.trainloader_CIFAR100
    testloader = data.testloader_CIFAR100
    fine_to_coarse = data.fine_to_coarse_CIFAR100
else:
    raise NotImplementedError

if args.model == "ResNet56":
    model = models.ResNet56(
        num_classes=len(trainloader.dataset.classes),  # type: ignore
    )  # type: ignore
elif args.model == "ResNet110":
    model = models.ResNet110(
        num_classes=len(trainloader.dataset.classes),  # type: ignore
    )
else:
    raise NotImplementedError

if args.loss == "XE":
    criterions = [losses.XE()]
elif args.loss == "XE,CE":
    criterions = [losses.XE(), losses.CE()]
elif args.loss == "GCE":
    criterions = [losses.GCE()]
elif args.loss == "XE,HCE":
    criterions = [losses.XE(), losses.HCE(fine_to_coarse)]  # type: ignore
elif args.loss == "HGCE":
    criterions = [losses.HGCE(fine_to_coarse)]  # type: ignore
else:
    raise NotImplementedError

if args.epochs > 0:
    epochs = args.epochs
else:
    raise ValueError

tag = f"{args.tag}_" if args.tag else args.tag
name = f"{tag}{args.model}_{args.dataset}_{args.loss}"

# Optimizers and Lr_schedulers hyperparams are fixed (no hyperparams search)
# and follow the ones proposed in Che+19 papers.
criterions: list[torch.nn.Module] = criterions
optimizers: list[torch.optim.Optimizer] = [
    torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    for _ in criterions
]
lr_schedulers: list[torch.optim.lr_scheduler._LRScheduler] = [
    torch.optim.lr_scheduler.MultiStepLR(o, milestones=[100, 150], gamma=0.1)
    for o in optimizers
]

# TRAIN ################################################################################

trainer = Trainer(
    name=name,
    model=model,
    criterions=criterions,
    optimizers=optimizers,
    lr_schedulers=lr_schedulers,
    epochs=epochs,
    trainloader=trainloader,
    testloader=testloader,
)

trainer.train()
