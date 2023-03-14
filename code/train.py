import argparse
import pathlib
from datetime import datetime

import data
import models
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

# PATHS -------------------------------------------------------------------------------

here = pathlib.Path(__file__).parent
now = datetime.now().strftime("%Y%m%dT%H%M%S")
models_path = here / "models"
runs_path = here / "runs"

models_path.mkdir(exist_ok=True)
runs_path.mkdir(exist_ok=True)

# CLI ---------------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Train model on CIFAR-10.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    epilog="Source code: https://github.com/S1M0N38/master-thesis",
)
parser.add_argument(
    "-q",
    "--quiet",
    action="store_true",
    default=False,
    help="Disable progress bar.",
)
parser.add_argument(
    "-m",
    "--model",
    action="store",
    type=str,
    choices=["SimpleCNN", "ResNet10"],
    required=True,
    help="Model to train.",
)
parser.add_argument(
    "-e",
    "--epochs",
    action="store",
    type=int,
    default=1,
    help="Number of epochs.",
)
parser.add_argument(
    "-lr",
    "--learning-rate",
    action="store",
    type=float,
    default=1e-3,
    help="Learning rate.",
)
parser.add_argument(
    "-v",
    "--val-per-epoch",
    action="store",
    type=int,
    default=5,
    help="Validation loops per epoch.",
)
parser.add_argument(  # TODO: implement
    "-r",
    "--resume",
    action="store",
    type=pathlib.Path,
    default=None,
    help="Resume training of a model.",
)
parser.add_argument(
    "-d",
    "--device",
    action="store",
    type=str,
    choices=["cpu", "cuda"],
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="Device used for training.",
)
parser.add_argument(
    "-t",
    "--tag",
    action="store",
    type=str,
    default="",
    help="Add tag to model name.",
)

args = parser.parse_args()

if args.resume:
    raise NotImplementedError("Resume from checkpoint is not implemented yet.")

summary = (
    "\nTraining parameters:\n"
    f"  - model: {args.model}\n"
    f"  - epochs: {args.epochs}\n"
    f"  - learning-rate: {args.learning_rate}\n"
    f"  - device: {args.device}\n"
)
print(summary)

# TRAINING AND VALIDATION LOOPS -------------------------------------------------------

# Auxilary var
VAL = len(data.trainloader) // args.val_per_epoch
args.tag = f"_{args.tag}" if args.tag else ""

model = getattr(models, args.model)().to(args.device)
model_name = f"{now}_{model.__class__.__name__}{args.tag}"

# Standard Loss and Optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

# Track experiments with TensorBoard
writer = SummaryWriter(log_dir=runs_path / model_name)

model.train()  # Set model in train mode
for epoch in range(args.epochs):
    running_loss = 0.0
    running_correct = 0

    # Train loop
    desc = f"Epoch [{epoch + 1:>2d}/{args.epochs}]: train"
    pbar = tqdm(data.trainloader, desc=desc, ncols=0, disable=args.quiet)

    for batch, (inputs, labels) in enumerate(pbar):
        # Send data to args.device
        inputs, labels = inputs.to(args.device), labels.to(args.device)

        # Forward propagation
        outputs, _ = model(inputs)  # make model prediction
        loss = criterion(outputs, labels)  # calculate loss
        correct = (outputs.argmax(1) == labels).sum()  # correct classificaiton

        # Backward propagation
        optimizer.zero_grad()  # reset gradient
        loss.backward()  # calculate gradient
        optimizer.step()  # update weights using gradient

        running_loss += loss.item()
        running_correct += correct.item()

        if (batch + 1) % VAL == 0:
            running_vloss = 0.0
            running_vcorrect = 0

            # Validation
            desc = f"Batch [{(batch + 1)//VAL:>2d}/{args.val_per_epoch}]: test"
            vpbar = tqdm(data.testloader, desc=desc, leave=False, disable=args.quiet)
            model.eval()  # Set model in eval mode

            with torch.no_grad():
                for vinputs, vlabels in vpbar:
                    vinputs, vlabels = vinputs.to(args.device), vlabels.to(args.device)
                    voutputs, _ = model(vinputs)
                    vloss = criterion(voutputs, vlabels)
                    vcorrect = (voutputs.argmax(1) == vlabels).sum()

                    running_vloss += vloss.item()
                    running_vcorrect += vcorrect.item()

            # Average Metrics from previous batches (loss per sample and accuracy)
            avg_loss = running_loss / (VAL * data.BATCH_SIZE)
            avg_vloss = running_vloss / len(data.testset)
            avg_accuracy = running_correct / (VAL * data.BATCH_SIZE)
            avg_vaccuracy = running_vcorrect / len(data.testset)

            # Add metrics to TensorBoard
            writer.add_scalars(
                "Loss",
                {"Training": avg_loss, "Validation": avg_vloss},
                epoch * len(data.trainloader) + batch,
            )
            writer.add_scalars(
                "Accuracy",
                {"Training": avg_accuracy, "Validation": avg_vaccuracy},
                epoch * len(data.trainloader) + batch,
            )

            # Add metrics to progress bar
            metrics = {
                "loss": avg_loss,
                "vloss": avg_vloss,
                "acc": avg_accuracy,
                "vacc": avg_vaccuracy,
            }
            pbar.set_postfix(metrics)

            # Reset state for next train step
            running_loss = 0.0
            running_correct = 0
            model.train()  # Set model back to train mode

# SAVE --------------------------------------------------------------------------------

checkpoint = {
    "args": args,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
}
torch.save(checkpoint, models_path / f"{model_name}.pth")
print(f"Checkpoint saved at: {model_name}.pth")
writer.flush()
writer.close()
