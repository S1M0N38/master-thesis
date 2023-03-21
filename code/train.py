import argparse

import data
import losses
import models
import torch
from trainer import Trainer

torch.manual_seed(0)

# CLI ##################################################################################

parser = argparse.ArgumentParser(
    description="Train pytorch models.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    epilog="Source code: https://github.com/S1M0N38/master-thesis",
)
parser.add_argument(
    "-t",
    "--trainer",
    type=str,
    action="store",
    choices=[
        "Che19a_XE",
        "Che19a_XE_CE",
        "Che19b_XE",
        "Che19b_GCE",
    ],
    required=True,
    help="Choose the Trainer to initialize and train",
)

# TODO: gpu selection is not working
#
# parser.add_argument(
#     "-g",
#     "--gpu",
#     type=int,
#     action="store",
#     choices=[0, 1],
#     default=0,
#     help="Select the GPU device to use. 0 for the first GPU, 1 for the second GPU.",
# )

args = parser.parse_args()

# TRAIN ################################################################################

if args.trainer == "Che19a_XE":
    model = models.ResNet110()
    criterion = losses.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=0.0001,
    )
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[100, 150],
        gamma=0.1,
    )
    trainer = Trainer(
        model=model,
        criterions=[criterion],
        optimizers=[optimizer],
        lr_schedulers=[lr_scheduler],
        epochs=200,
        trainloader=data.trainloader_CIFAR10,
        testloader=data.testloader_CIFAR10,
        save_epoch=10,
        tag=args.trainer,
    )

elif args.trainer == "Che19a_XE_CE":
    model = models.ResNet110()
    criterion_xe = losses.CrossEntropyLoss()
    optimizer_xe = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=0.0001,
    )
    lr_scheduler_xe = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_xe,
        milestones=[100, 150],
        gamma=0.1,
    )
    criterion_ce = losses.ComplementEntropyLoss()
    optimizer_ce = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=0.0001,
    )
    lr_scheduler_ce = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_ce,
        milestones=[100, 150],
        gamma=0.1,
    )
    trainer = Trainer(
        model=model,
        criterions=[criterion_xe, criterion_ce],
        optimizers=[optimizer_xe, optimizer_ce],
        lr_schedulers=[lr_scheduler_xe, lr_scheduler_ce],
        epochs=200,
        trainloader=data.trainloader_CIFAR10,
        testloader=data.testloader_CIFAR10,
        save_epoch=10,
        tag=args.trainer,
    )

elif args.trainer == "Che19b_XE":
    model = models.ResNet56()
    criterion = losses.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=0.0001,
    )
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[100, 150],
        gamma=0.1,
    )
    trainer = Trainer(
        model=model,
        criterions=[criterion],
        optimizers=[optimizer],
        lr_schedulers=[lr_scheduler],
        epochs=200,
        trainloader=data.trainloader_CIFAR10,
        testloader=data.testloader_CIFAR10,
        save_epoch=10,
        tag=args.trainer,
    )

elif args.trainer == "Che19b_GCE":
    model = models.ResNet56()
    criterion = losses.GuidedComplementEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=0.0001,
    )
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[100, 150],
        gamma=0.1,
    )
    trainer = Trainer(
        model=model,
        criterions=[criterion],
        optimizers=[optimizer],
        lr_schedulers=[lr_scheduler],
        epochs=200,
        trainloader=data.trainloader_CIFAR10,
        testloader=data.testloader_CIFAR10,
        save_epoch=10,
        tag=args.trainer,
    )

trainer.train()  # type: ignore
