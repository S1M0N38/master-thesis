import argparse
import pathlib

import data
import models
import torch
import torchattacks
import torchvision.transforms as transforms
from torch.utils.tensorboard.writer import SummaryWriter
from torchmetrics import Accuracy
from tqdm import tqdm

PATH_HERE = pathlib.Path(__file__).parent
PATH_MODELS = PATH_HERE / "models"
PATH_RUNS = PATH_HERE / "runs"

# CLI ##################################################################################

parser = argparse.ArgumentParser(
    description="Choose which Model to attack.",
    epilog="Source code: https://github.com/S1M0N38/master-thesis",
)
parser.add_argument(
    "-l",
    "--list-models",
    action="store_true",
    help="List all available pytorch models.",
)
parser.add_argument(
    "-m",
    "--model",
    type=pathlib.Path,
    help="Select a model to use.",
)
parser.add_argument(
    "-s",
    "--save",
    action="store_true",
    help="Save clean and adversarial images from first batch in TensorBoard.",
)
args = parser.parse_args()

if args.list_models:
    print(f"Available models in {PATH_MODELS}:")
    for file in PATH_MODELS.glob("*.pth"):
        print(file.relative_to(PATH_HERE))
    exit()

if args.model is None:
    parser.print_help()
    exit()

if not args.model.exists():
    raise FileNotFoundError(
        f"Model file {args.model} not found.\n"
        "Use -l to get the list of available models."
    )

# ATTACK ###############################################################################

# Set up the experiment
writer = SummaryWriter(log_dir=PATH_RUNS / args.model.stem)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Example format for args.models.stem
# {DATE}-{TIME}_{ARCHITECTURE}_{REF}_{LOSS}.pth

architecture = getattr(models, str(args.model.stem).split("_")[1])
model = architecture().to(device)
checkpoint = torch.load(args.model, map_location=device)
model.load_state_dict(checkpoint["model"])

direct_transform = transforms.Normalize(
    mean=(0.4914, 0.4822, 0.4465),
    std=(0.2023, 0.1994, 0.2010),
)
inverse_transform = transforms.Normalize(
    mean=(-0.4914 / 0.2023, -0.4822 / 0.1994, -0.4465 / 0.2010),
    std=(1 / 0.2023, 1 / 0.1994, 1 / 0.2010),
)

attack = torchattacks.FGSM(model, eps=8 / 255)

accuracy = Accuracy("multiclass", num_classes=10).to(device)
accuracy_adv = Accuracy("multiclass", num_classes=10).to(device)

testloader = tqdm(data.testloader_CIFAR10)

for batch, (inputs, labels) in enumerate(testloader):
    inputs, labels = inputs.to(device), labels.to(device)

    # Clean Input Accuracy
    outputs = model(inputs)
    accuracy(outputs, labels)

    # Adversarial Input Accuracy
    inputs_adv = attack(inverse_transform(inputs), labels)
    outputs_adv = model(direct_transform(inputs_adv))
    accuracy_adv(outputs_adv, labels)

    # Add images from the first batch to TensorBoard
    if batch == 0 and args.save:
        writer.add_images("Inputs", inverse_transform(inputs))
        writer.add_images("Inputs Adversarial", inputs_adv)

    # Only first batch when using CPU
    if device == "cpu":
        break

print(f"Test Clean Accuracy: {accuracy.compute().item():.4f}")
print(f"Test Adversarial Accuracy: {accuracy_adv.compute().item():.4f}")
