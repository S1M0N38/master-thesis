import math

import torch
import torch.nn.functional as F
from torch import nn

# BUILTIN LOSSES #######################################################################

CrossEntropyLoss = nn.CrossEntropyLoss


# ORIGINAL IMPLEMENTATIONS #############################################################


class _ComplementEntropyLoss(nn.Module):
    """
    Original implementation of 'Complement Entropy Loss'
    https://github.com/henry8527/COT/blob/master/code/COT.py

    Note: The rescaling factor has been changed from self.classes to self.classes - 1 to
        ensure consistency with the original paper.
    """

    def __init__(self):
        super().__init__()

    # here we implemented step by step for corresponding to our formula
    # described in the paper
    def forward(self, yHat, y):
        self.batch_size, self.classes = y_hat.shape

        yHat = F.softmax(yHat, dim=1)
        Yg = torch.gather(yHat, 1, torch.unsqueeze(y, 1))
        Yg_ = (1 - Yg) + 1e-7  # avoiding numerical issues (first)
        Px = yHat / Yg_.view(len(yHat), 1)
        Px_log = torch.log(Px + 1e-10)  # avoiding numerical issues (second)
        y_zerohot = torch.ones(self.batch_size, self.classes).scatter_(
            1, y.view(self.batch_size, 1).data.cpu(), 0
        )
        output = Px * Px_log * y_zerohot
        loss = torch.sum(output)
        loss /= float(self.batch_size)
        loss /= float(self.classes - 1)
        return loss


class _GuidedComplementEntropyLoss(nn.Module):
    """
    Original implementation of 'Guided Complement Entropy Loss'
    https://github.com/henry8527/GCE/blob/master/GCE.py

    Note: The rescaling factor has been changed from self.classes to self.classes - 1 to
        ensure consistency with the original paper.
    """

    def __init__(self, alpha=0.2):
        super().__init__()
        self.alpha = alpha

    # here we implemented step by step for corresponding to our formula
    # described in the paper
    def forward(self, yHat, y):
        self.batch_size, self.classes = y_hat.shape

        yHat = F.softmax(yHat, dim=1)
        Yg = torch.gather(yHat, 1, torch.unsqueeze(y, 1))
        Yg_ = (1 - Yg) + 1e-7  # avoiding numerical issues (first)
        # avoiding numerical issues (second)
        guided_factor = (Yg + 1e-7) ** self.alpha
        Px = yHat / Yg_.view(len(yHat), 1)
        Px_log = torch.log(Px + 1e-10)  # avoiding numerical issues (third)
        y_zerohot = torch.ones(self.batch_size, self.classes).scatter_(
            1, y.view(self.batch_size, 1).data.cpu(), 0
        )
        output = Px * Px_log * y_zerohot
        guided_output = guided_factor.squeeze() * torch.sum(output, dim=1)
        loss = torch.sum(guided_output)
        loss /= float(self.batch_size)
        loss /= math.log(float(self.classes - 1))
        return loss


# CUSTOM IMPLEMENTATIONS ###############################################################


class GuidedComplementEntropyLoss(nn.Module):
    """
    Computes the Guided Complement Entropy loss between the predicted scores and the
    target labels, as described in the paper "Improving Adversarial Robustness via
    Guided Complement Entropy" by Chen et al. (2019).

    Args:
        alpha (float, optional): The exponential factor applied to the guided factor.
            Default: 0.2
        reduction (str, optional): Specifies the reduction to apply to the output.
            Valid options are 'mean', 'sum', and 'none'. Default: 'mean'
        standard (bool, optional): If True, standardize the loss by dividing it by the
            standard_func of the number of classes minus one. Default: True
        standard_func (function, optional): The function used in the standardization
            factor. Default: math.log

    Shapes:
        - input: (batch_size, num_classes) where num_classes is the number of classes in
            the dataset
        - target: (batch_size,) where each element is an integer representing the class
            index

    Returns:
        torch.Tensor: A scalar tensor representing the computed loss.

    Example:
        >>> loss_fn = GuidedComplementEntropyLoss()
        >>> input = torch.randn(3, 5)
        >>> target = torch.tensor([2, 4, 0])
        >>> loss = loss_fn(input, target)
    """

    def __init__(
        self,
        alpha: float = 0.2,
        reduction: str = "mean",
        standard: bool = True,
        standard_func=math.log,
    ):
        super().__init__()

        # Exp for the guided factor
        self.alpha = alpha

        # Reduction operator
        if reduction == "mean":
            self.reduction = torch.mean
        elif reduction == "sum":
            self.reduction = torch.sum
        elif reduction == "none":
            self.reduction = lambda x: x
        else:
            raise ValueError(f"'{reduction}' is not a valid reduction operator")

        # Normalization Factor
        self.standard = standard
        self.standard_func = standard_func

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the Guided Complement Entropy loss.

        Args:
            input (torch.Tensor): The input tensor of shape (batch_size, num_classes).
            target (torch.Tensor): The target tensor of shape (batch_size,).

        Returns:
            torch.Tensor: A scalar tensor representing the computed loss.
        """
        # Batch and Class dimensions
        B, K = input.shape

        y_one_hot = F.one_hot(target, num_classes=K).bool()

        # Get g-component normalilzed
        y_hat = F.softmax(input, dim=1)
        y_hat_g = torch.masked_select(y_hat, y_one_hot).reshape(B, 1)
        # Remove g-component and normalilze (wo = without)
        z_wo_g = torch.masked_select(input, ~y_one_hot).reshape(B, K - 1)
        y_hat_wo_g = F.softmax(z_wo_g, dim=1)

        # Numerical stability for log
        y_hat_wo_g = torch.clamp(y_hat_wo_g, min=1e-30)

        # Minus Shannon Entropy with guided factor
        guided_factor = (y_hat_g.squeeze() + 1e-7) ** self.alpha
        loss = guided_factor * torch.sum(y_hat_wo_g * torch.log(y_hat_wo_g), dim=-1)

        # Reduction and Standardization
        loss = self.reduction(loss)
        loss = loss / self.standard_func(K - 1) if self.standard else loss

        return loss


class ComplementEntropyLoss(GuidedComplementEntropyLoss):
    """
    Complement Entropy Loss is a variant of Guided Complement Entropy Loss,
    where the guided factor is set to zero. It was proposed in the paper
    "Complement Objective Training" by Chen et al. (2019).

    Args:
        reduction (str, optional): Specifies the reduction to apply to the
            output. Options are 'mean', 'sum', or 'none'. Default is 'mean'.
        standard (bool, optional): Specifies whether to standardize the output
            loss by dividing it by standard_func(K-1), where K is the number of
            classes. Default is True.
        standard_func (function, optional): Specifies the function used to
            calculate the normalization factor. Default is identity function.
    """

    def __init__(
        self,
        reduction: str = "mean",
        standard: bool = True,
        standard_func=lambda x: x,
    ):
        super().__init__(
            reduction=reduction,
            standard=standard,
            standard_func=standard_func,  # type: ignore
            alpha=0,
        )


# TESTS AND BENCHMARKS #################################################################

if __name__ == "__main__":
    import torch.utils.benchmark as benchmark

    # TODO: use multiple batches for benchmarking
    # Benchmarks runs
    runs = 1000

    # Bactch size and number of classes
    B, K = 128, 10

    # Dummy inputs for benchmark (if *100 _ce and ce differs)
    y_hat = torch.rand(B, K) * 10
    y = torch.randint(K, (B,))

    losses = (
        (_ComplementEntropyLoss(), ComplementEntropyLoss()),
        (_GuidedComplementEntropyLoss(), GuidedComplementEntropyLoss()),
    )

    for _loss, loss in losses:
        assert _loss(y_hat, y).allclose(loss(y_hat, y))

        _results = benchmark.Timer(
            setup=(
                f"from __main__ import {_loss.__class__.__name__};" f"_loss = {_loss}"
            ),
            stmt="_loss(y_hat, y)",
            globals={"y_hat": y_hat, "y": y},
        ).timeit(runs)

        results = benchmark.Timer(
            setup=(f"from __main__ import {loss.__class__.__name__};" f"loss = {loss}"),
            stmt="loss(y_hat, y)",
            globals={"y_hat": y_hat, "y": y},
        ).timeit(runs)

        print(f"\n \033[31m{_loss.__class__.__name__}\033[0m\n {_results}")
        print(f"\n \033[32m{loss.__class__.__name__}\033[0m\n {results} \n")
        print("-" * 88)
