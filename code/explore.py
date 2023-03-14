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
    description="Explore model with TensorBoard (architecture and embeddings).",
    epilog="Source code: https://github.com/S1M0N38/master-thesis",
)
parser.add_argument(  # TODO: test
    "-c",
    "--checkpoint",
    action="store",
    type=pathlib.Path,
    default=sorted(models_path.glob("*.pth"))[-1],
    help="Model checkpoint to explore. Default last trained model.",
)
parser.add_argument(
    "-d",
    "--device",
    action="store",
    type=str,
    choices=["cpu", "cuda"],
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="Device used for exploring. Default best device avaiable",
)
parser.add_argument(
    "-e",
    "--embedding-samples",
    action="store",
    type=int,
    default=1000,
    help="Number of samples used to embedding visualization.",
)


args = parser.parse_args()

assert args.checkpoint.is_file()
assert args.checkpoint.suffix == ".pth"


# LOAD MODEL ---------------------------------------------------------------------------

model_name = args.checkpoint.stem.split("_")[1]
model = getattr(models, model_name)().to(args.device)
checkpoint = torch.load(args.checkpoint, map_location=args.device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Visualization with TensorBoard
writer = SummaryWriter(log_dir=str(runs_path / args.checkpoint.stem))

# EXPLORE ARCHITECTURE -----------------------------------------------------------------

## Not supported for ResNet10
# inputs, labels = next(iter(data.trainloader))
# inputs, labels = inputs.to(args.device), labels.to(args.device)
# writer.add_graph(model, inputs)
# writer.flush()


# EXPLORE EMBEDDINGS -------------------------------------------------------------------

labels = []
outputs = []

pbar = tqdm(data.trainloader, total=args.embedding_samples // data.BATCH_SIZE)
for batch, (binputs, blabels) in enumerate(pbar):
    if data.BATCH_SIZE * batch > args.embedding_samples:
        break

    binputs, blabels = binputs.to(args.device), blabels.to(args.device)
    boutputs, bfeatures = model(binputs)

    labels.append(blabels)
    outputs.append(boutputs)

labels = torch.cat(labels)
outputs = torch.cat(outputs)

class_labels = [data.classes[label] for label in labels]
writer.add_embedding(outputs, metadata=class_labels)
writer.flush()
writer.close()
