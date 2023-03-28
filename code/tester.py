import pathlib
from datetime import datetime
from typing import Callable

import torch
from metrics import LCA
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, MetricCollection
from tqdm import tqdm

NOW = datetime.utcnow().strftime("%m%d-%H%M")

PATH_HERE = pathlib.Path(__file__).parent
PATH_MODELS = PATH_HERE / "models"
PATH_RUNS = PATH_HERE / "runs"
PATH_MODELS.mkdir(exist_ok=True)
PATH_RUNS.mkdir(exist_ok=True)


class Tester:
    # TODO: add docstring
    def __init__(
        self,
        model: torch.nn.Module,
        testloader: DataLoader,
        lca: torch.Tensor,
        attacks: dict[str, Callable | None] = {},
        pre_transform: Callable = lambda x: x,
        post_transform: Callable = lambda x: x,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.device = device
        self.model = model.to(self.device)
        self.model.eval()

        # attack
        self.attacks = attacks
        self.attacks["Baseline"] = None

        # Data
        self.testloader = testloader
        self.num_classes = len(self.testloader.dataset.classes)  # type: ignore
        self.lca = lca.to(self.device)
        self.pre_transform = pre_transform
        self.post_transform = post_transform

        # Metrics
        self.metrics = {
            name: MetricCollection(
                [
                    Accuracy("multiclass", num_classes=self.num_classes).to(device),
                    LCA(self.lca).to(device),
                ]
            )
            for name in self.attacks.keys()
        }

    def test(self) -> dict[str, MetricCollection]:
        metrics = {}
        for i, (name, attack) in enumerate(self.attacks.items()):
            desc = f"Test [{i + 1:>2d}/{len(self.attacks)}]: {name}"
            testloader = tqdm(self.testloader, desc=desc, disable=True)

            for inputs, labels in testloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                if attack is not None:
                    inputs = self.pre_transform(inputs)
                    inputs = attack(inputs, labels)
                    inputs = self.post_transform(inputs)

                outputs = self.model(inputs)
                self.metrics[name](outputs, labels)

            metrics[name] = self.metrics[name].compute()
        return metrics
