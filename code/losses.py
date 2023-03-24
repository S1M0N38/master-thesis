import math

import torch
import torch.nn.functional as F
from torch import nn

# TODO: move to a simpler Naming Scheme for Loss

# BUILTIN LOSSES #######################################################################


class XE(nn.CrossEntropyLoss):
    def __init__(self):
        super().__init__()


# ORIGINAL IMPLEMENTATIONS #############################################################


class _CE(nn.Module):
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


class _GCE(nn.Module):
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


class _InnerComplementEntropy(nn.Module):
    def __init__(self, fine2coarse):
        super().__init__()
        self.fine2coarse = torch.Tensor(fine2coarse).long()

    def forward(self, yHat, y_fine):
        self.batch_size = len(y_fine)
        self.coarse_classes = 5

        y_coarse = torch.unsqueeze(self.fine2coarse[y_fine], 1)
        y_G = torch.topk((self.fine2coarse == y_coarse).int(), self.coarse_classes)[1]
        yHat_G = F.softmax(torch.gather(yHat, 1, y_G).float(), dim=1)
        new_Yg_index = torch.topk((y_G == torch.unsqueeze(y_fine, 1)).int(), 1)[1]
        Yg = torch.gather(yHat_G, 1, new_Yg_index)
        Yg_ = (1.0 - Yg) + 1e-7  # avoiding numerical issues (first)
        Px = yHat_G / Yg_.view(len(yHat_G), 1)
        Px_log = torch.log(Px.clamp(min=1e-10))  # avoiding numerical issues (second)
        y_zerohot = torch.ones(self.batch_size, self.coarse_classes).scatter_(
            1, new_Yg_index.data.cpu(), 0
        )
        output = Px * Px_log * y_zerohot
        loss = torch.sum(output)
        loss /= float(self.batch_size)
        loss /= float(self.coarse_classes - 1)
        return loss


class _OuterComplementEntropy(nn.Module):
    def __init__(self, fine2coarse):
        super().__init__()
        self.fine2coarse = torch.Tensor(fine2coarse).long()

    def forward(self, yHat, y_fine):
        self.batch_size = len(y_fine)
        self.fine_classes = 100
        yHat = F.softmax(yHat, dim=1)
        y_fine = y_fine.long()
        y_coarse = torch.unsqueeze(self.fine2coarse[y_fine], 1)
        y_G = torch.topk((self.fine2coarse == y_coarse).int(), 5)[1]
        Yg = torch.sum(torch.gather(yHat, 1, y_G), dim=1)
        Yg_ = (1.0 - Yg) + 1e-7  # avoiding numerical issues (first)
        Px = yHat / Yg_.view(len(yHat), 1)
        Px_log = torch.log(Px.clamp(min=1e-10))  # avoiding numerical issues (second)
        y_zerohot = torch.ones(self.batch_size, self.fine_classes).scatter_(
            1, y_G.data.cpu(), 0
        )
        output = Px * Px_log * y_zerohot
        loss = torch.sum(output, dim=-1)
        loss = loss.sum()
        loss /= float(self.batch_size)
        loss /= float(self.fine_classes - 5)
        return loss


class _HCE(nn.Module):
    def __init__(self, fine2coarse):
        super().__init__()

        self.fine2coarse = fine2coarse
        self._ICE = _InnerComplementEntropy(fine2coarse)
        self._OCE = _OuterComplementEntropy(fine2coarse)

    def forward(self, yHat, y_fine):
        return self._OCE(yHat, y_fine) + self._ICE(yHat, y_fine)


# CUSTOM IMPLEMENTATIONS ###############################################################


class GCE(nn.Module):
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
        >>> loss_fn = GCE()
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
        y_hat_wo_g = torch.clamp(y_hat_wo_g, min=1e-20)

        # Minus Shannon Entropy with guided factor
        guided_factor = (y_hat_g.squeeze() + 1e-7) ** self.alpha
        loss = guided_factor * torch.sum(y_hat_wo_g * torch.log(y_hat_wo_g), dim=-1)

        # Reduction and Standardization
        loss = self.reduction(loss)
        loss = loss / self.standard_func(K - 1) if self.standard else loss

        return loss


class CE(GCE):
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


class HGCE(nn.Module):
    def __init__(
        self,
        fine_to_coarse: list[int],
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
        self.num_fine_classes = len(fine_to_coarse)
        self.num_coarse_classes = max(fine_to_coarse) + 1
        assert self.num_fine_classes % self.num_coarse_classes == 0

        device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.coarse_masks = (
            torch.zeros((self.num_fine_classes, self.num_fine_classes))
            .bool()
            .to(device)
        )
        for i, label1 in enumerate(fine_to_coarse):
            for j, label2 in enumerate(fine_to_coarse):
                if label1 == label2:
                    self.coarse_masks[i, j] = True

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the Guided Complement Entropy loss.

        Args:
            input (torch.Tensor): The input tensor of shape (batch_size, num_classes).
            target (torch.Tensor): The target tensor of shape (batch_size, num_hlevels).

        Returns:
            torch.Tensor: A scalar tensor representing the computed loss.
        """

        # Batches (B), Fine Classes (K0), Coarse Classes (K1)
        # Number of Fine Classes for each Coarse Class (K01)
        B, K0 = input.shape
        K0 = self.num_fine_classes
        K1 = self.num_coarse_classes
        K01 = K0 // K1

        # One-hot encoding for fine labels
        y0 = F.one_hot(target, num_classes=K0).bool()

        # One-hot encoding for coarse labels
        y1 = self.coarse_masks[target]

        # Get g-component normalilzed
        y_hat = F.softmax(input, dim=1)
        y_hat_g = torch.masked_select(y_hat, y0).reshape(B, 1)

        # Remove G-components, normalilze, and calc loss_K_wo_G
        z_wo_G = torch.masked_select(input, ~y1).reshape(B, K0 - K01)
        y_hat_wo_G = F.softmax(z_wo_G, dim=1)
        y_hat_wo_G = torch.clamp(y_hat_wo_G, min=1e-20)
        loss_K_wo_G = torch.sum(y_hat_wo_G * torch.log(y_hat_wo_G), dim=-1)
        if self.standard:
            loss_K_wo_G /= self.standard_func(K0 - K01)

        # Remove select G and remove g-components, normalilze, and calc loss_G_wo_g
        z_G_wo_g = torch.masked_select(input, y1 & ~y0).reshape(B, K01 - 1)
        y_hat_G_wo_g = F.softmax(z_G_wo_g, dim=1)
        y_hat_G_wo_g = torch.clamp(y_hat_G_wo_g, min=1e-20)
        loss_G_wo_g = torch.sum(y_hat_G_wo_g * torch.log(y_hat_G_wo_g), dim=-1)
        if self.standard:
            loss_G_wo_g /= self.standard_func(K01 - 1)

        # Sum intra-G and extra-G losses and scale by guided factor
        guided_factor = (y_hat_g.squeeze() + 1e-7) ** self.alpha
        loss = guided_factor * (loss_G_wo_g + loss_K_wo_G)

        # Reduction
        loss = self.reduction(loss)

        return loss


class HCE(HGCE):
    def __init__(
        self,
        fine_to_coarse: list[int],
        reduction: str = "mean",
        standard: bool = True,
        standard_func=lambda x: x,
    ):
        super().__init__(
            fine_to_coarse=fine_to_coarse,
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
    B, K = 128, 100

    # Dummy inputs for benchmark (if *100 _ce and ce differs)
    y_hat = torch.rand(B, K) * 100
    y = torch.randint(K, (B,)).long()

    losses = (
        (_CE(), CE()),
        (_GCE(), GCE()),
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

    # Test Hierarchical Complement Entropy
    # fmt: off
    fine_to_coarse_CIFAR100 = [
        4,  1, 14,  8,  0,  6,  7,  7, 18,  3,  3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
        6, 11,  5, 10,  7,  6, 13, 15,  3, 15,  0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
        5, 19,  8,  8, 15, 13, 14, 17, 18, 10, 16,  4, 17,  4,  2,  0, 17,  4, 18, 17,
        10,  3,  2, 12, 12, 16, 12,  1,  9, 19,  2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
        16, 19,  2,  4,  6, 19,  5,  5,  8, 19, 18,  1,  2, 15,  6,  0, 17,  8, 14, 13,
    ]
    # fmt: on
    _hce = _HCE(fine_to_coarse_CIFAR100)
    hce = HCE(fine_to_coarse_CIFAR100)

    print(_hce(y_hat, y))
    print(hce(y_hat, y))
    assert _hce(y_hat, y).allclose(hce(y_hat, y))
