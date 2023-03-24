import pathlib
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchmetrics import Accuracy, MeanMetric
from tqdm import tqdm

NOW = datetime.utcnow().strftime("%m%d-%H%M")

PATH_HERE = pathlib.Path(__file__).parent
PATH_MODELS = PATH_HERE / "models"
PATH_RUNS = PATH_HERE / "runs"
PATH_MODELS.mkdir(exist_ok=True)
PATH_RUNS.mkdir(exist_ok=True)

# TODO: fix docs for Trainer


class Trainer:
    """
    A class to train a PyTorch model with multiple optimizers and schedulers, and track
    training and validation metrics. Multiple steps are performed sequentially during
    training and validation.

    Args:
        model (torch.nn.Module): The PyTorch model to be trained.
        criterions (List[torch.nn.Module]): The loss functions to be used during
            training.
        optimizers (List[torch.optim.Optimizer]): The optimizers to be used during
            training.
        lr_schedulers (List[torch.optim.lr_scheduler._LRScheduler]):
            The learning rate schedulers to be used during training.
        epochs (int): The number of training epochs.
        trainloader (DataLoader): The data loader for the training dataset.
        testloader (DataLoader): The data loader for the validation dataset.
        device (str, optional): The device to run the training on.
            Defaults to "cuda" if CUDA is available, else "cpu".
        val_per_epoch (int, optional): The number of times to perform validation per
            epoch. Defaults to 5.
        save_epoch (int or None, optional): The number of epochs after which to save a
            model checkpoint. If None, the model is not saved during training.
            Defaults to None.
        tag (str, optional): A tag to add to the name of the model, for TensorBoard
            logging purposes. Defaults to "".
    """

    def __init__(
        self,
        name: str,
        model: torch.nn.Module,
        criterions: list[torch.nn.Module],
        optimizers: list[torch.optim.Optimizer],
        lr_schedulers: list[torch.optim.lr_scheduler._LRScheduler],
        epochs: int,
        trainloader: DataLoader,
        testloader: DataLoader,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        val_per_epoch: int = 5,
        save_epoch: int | None = None,
    ) -> None:
        # Model
        self.name = name
        self.device = device
        self.model = model.to(self.device)

        # Train
        self.criterions = criterions
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        self.epochs = epochs

        # Data
        self.trainloader = trainloader
        self.testloader = testloader
        self.num_classes = len(self.trainloader.dataset.classes)  # type: ignore

        # Track experiments with TensorBoard
        self.writer = SummaryWriter(log_dir=PATH_RUNS / self.name, flush_secs=30)

        # Metrics
        self.metrics = [
            {
                "loss": MeanMetric().to(self.device),
                "vloss": MeanMetric().to(self.device),
                "acc": Accuracy("multiclass", num_classes=self.num_classes).to(
                    self.device
                ),
                "vacc": Accuracy("multiclass", num_classes=self.num_classes).to(
                    self.device
                ),
            }
            for _ in criterions
        ]

        # Perform validation every val_batch batches
        self.val_per_epoch = val_per_epoch
        self.val_batch = len(self.trainloader) // val_per_epoch

        # Save model every save_epoch epochs
        self.save_epoch = epochs if save_epoch is None else save_epoch

    def _train_epoch(self, epoch: int) -> None:
        desc = f"Epoch [{epoch + 1:>2d}/{self.epochs}]: train"
        trainloader = tqdm(self.trainloader, desc=desc, ncols=0)

        for batch, (inputs, labels) in enumerate(trainloader):
            # Send data to self.device
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            for criterion, optimizer, metrics in zip(
                self.criterions, self.optimizers, self.metrics
            ):
                # Forward propagation: make model prediction and calculate loss
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                # Update metrics: training loss and training accuracy
                metrics["loss"](loss.item() / self.trainloader.batch_size)
                metrics["acc"](outputs, labels)

                # Backward propagation: reset grad, calc grad, update weights using grad
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (batch + 1) % self.val_batch == 0:
                # Validate model and add metrics to TensorBoard
                self._validate(batch)
                metrics = self._metrics(epoch, batch)

        for lr_scheduler in self.lr_schedulers:
            lr_scheduler.step()

    def _validate(self, batch: int) -> None:
        desc = f"Batch [{(batch + 1)//self.val_batch:>2d}/{self.val_per_epoch}]: test"
        testloader = tqdm(self.testloader, desc=desc, leave=False)

        self.model.eval()
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                for criterion, metrics in zip(self.criterions, self.metrics):
                    # Forward propagation: make prediction and calc validation loss
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)

                    # Update metrics: validation loss and validation accuracy
                    metrics["vloss"](loss.item() / self.testloader.batch_size)
                    metrics["vacc"](outputs, labels)

        self.model.train()

    def _metrics(self, epoch: int, batch: int):
        for criterion, metrics in zip(self.criterions, self.metrics):
            # Compute metrics and reset metrics for the nexy validation cycle
            scalars = {}
            for key, value in metrics.items():
                scalars[key] = value.compute().item()
                value.reset()

            # Add metrics to tensorboard
            self.writer.add_scalars(
                f"{criterion.__class__.__name__}_Loss",
                {"Training": scalars["loss"], "Validation": scalars["vloss"]},
                epoch * len(self.trainloader) + batch,
            )
            self.writer.add_scalars(
                f"{criterion.__class__.__name__}_Accuracy",
                {"Training": scalars["acc"], "Validation": scalars["vacc"]},
                epoch * len(self.trainloader) + batch,
            )
            # Clean a bit TensorBoard
            # self.writer.add_scalars(
            #     f"{criterion.__class__.__name__}_Error",
            #     {"Training": 1 - scalars["acc"], "Validation": 1 - scalars["vacc"]},
            #     epoch * len(self.trainloader) + batch,
            # )

        self.writer.flush()

    def _save_checkpoint(self, epoch):
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizers": [o.state_dict() for o in self.optimizers],
            "lr_schedulers": [s.state_dict() for s in self.lr_schedulers],
        }
        path = PATH_MODELS / f"{self.name}_{epoch:0>4}.pth"
        torch.save(checkpoint, path)
        print(f"Checkpoint saved at {path.relative_to(PATH_HERE)}")

    def _load_checkpoint(self):
        raise NotImplementedError

    def train(self) -> None:
        """
        Trains the model for the specified number of epochs, using the specified
        data loaders, optimizer, and learning rate scheduler.
        """
        for epoch in range(self.epochs):
            self._train_epoch(epoch)
            self.writer.flush()
            if (epoch + 1) % self.save_epoch == 0:
                self._save_checkpoint(epoch + 1)
        self.writer.close()
