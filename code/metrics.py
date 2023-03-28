from typing import Optional

import torch
from torchmetrics import Metric


class LCA(Metric):
    higher_is_better: Optional[bool] = False

    # TODO: add docstring

    def __init__(self, lca):
        super().__init__()
        self.lca = lca
        self.add_state("distance", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("misclassified", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = preds.argmax(dim=-1)
        self.distance += self.lca[target, preds].sum()
        self.misclassified += target.numel() - (preds == target).sum()  # type: ignore

    def compute(self):
        return self.distance.float() / self.misclassified  # type: ignore
