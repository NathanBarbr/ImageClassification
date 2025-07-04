import torch
from typing import Tuple
from abc import ABCMeta, abstractmethod
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader


# for wandb users:
from assignment_1_code.wandb_logger import WandBLogger


class BaseTrainer(metaclass=ABCMeta):
    """
    Base class of all Trainers.
    """

    @abstractmethod
    def train(self) -> None:
        """
        Holds training logic.
        """

        pass

    @abstractmethod
    def _val_epoch(self) -> Tuple[float, float, float]:
        """
        Holds validation logic for one epoch.
        """

        pass

    @abstractmethod
    def _train_epoch(self) -> Tuple[float, float, float]:
        """
        Holds training logic for one epoch.
        """

        pass


class ImgClassificationTrainer(BaseTrainer):
    """
    Class that stores the logic for training a model for image classification.
    """

    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        lr_scheduler,
        train_metric,
        val_metric,
        train_data,
        val_data,
        device,
        num_epochs: int,
        training_save_dir: Path,
        batch_size: int = 4,
        val_frequency: int = 5,
    ) -> None:
        """
        Args and Kwargs:
            model (nn.Module): Deep Network to train
            optimizer (torch.optim): optimizer used to train the network
            loss_fn (torch.nn): loss function used to train the network
            lr_scheduler (torch.optim.lr_scheduler): learning rate scheduler used to train the network
            train_metric (dlvc.metrics.Accuracy): Accuracy class to get mAcc and mPCAcc of training set
            val_metric (dlvc.metrics.Accuracy): Accuracy class to get mAcc and mPCAcc of validation set
            train_data (dlvc.datasets.cifar10.CIFAR10Dataset): Train dataset
            val_data (dlvc.datasets.cifar10.CIFAR10Dataset): Validation dataset
            device (torch.device): cuda or cpu - device used to train the network
            num_epochs (int): number of epochs to train the network
            training_save_dir (Path): the path to the folder where the best model is stored
            batch_size (int): number of samples in one batch
            val_frequency (int): how often validation is conducted during training (if it is 5 then every 5th
                                epoch we evaluate model on validation set)

        What does it do:
            - Stores given variables as instance variables for use in other class methods e.g. self.model = model.
            - Creates data loaders for the train and validation datasets
            - Optionally use weights & biases for tracking metrics and loss: initializer W&B logger

        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler
        self.train_metric = train_metric
        self.val_metric = val_metric
        self.device = device
        self.num_epochs = num_epochs
        self.training_save_dir = training_save_dir
        self.batch_size = batch_size
        self.val_frequency = val_frequency
        self.best_pcacc = 0.0  # for tracking best validation per-class accuracy

        # Loaders
        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

        self.wandb_logger = WandBLogger(enabled=True, model=self.model)

        pass

    def _train_epoch(self, epoch_idx: int) -> Tuple[float, float, float]:
        """
        Training logic for one epoch.
        Prints current metrics at end of epoch.
        Returns loss, mean accuracy and mean per class accuracy for this epoch.

        epoch_idx (int): Current epoch number
        """
        self.model.train()
        self.train_metric.reset()

        total_loss = 0.0
        num_batches = 0

        for x, y in tqdm(self.train_loader, desc=f"Train Epoch {epoch_idx}"):
            x, y = x.to(self.device), y.to(self.device)

            # Forward
            out = self.model(x)
            loss = self.loss_fn(out, y)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Logging
            self.train_metric.update(out, y)
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        acc = self.train_metric.accuracy()
        pc_acc = self.train_metric.per_class_accuracy()

        print(f"\n[Train] Epoch {epoch_idx}")
        print(f"\n______epoch {epoch_idx}\n")
        print(f"accuracy: {acc:.4f}")
        print(f"per class accuracy: {pc_acc:.4f}")


        for class_idx, classname in enumerate(self.train_metric.classes):
            total = self.train_metric.total_pred[classname]
            correct = self.train_metric.correct_pred[classname]
            if total > 0:
                acc_cls = correct / total
                print(f"Accuracy for class: {classname:<6} is {acc_cls:.2f}")
            else:
                print(f"Accuracy for class: {classname:<6} is N/A")

        return avg_loss, acc, pc_acc


    def _val_epoch(self, epoch_idx: int) -> Tuple[float, float, float]:
        """
        Validation logic for one epoch.
        Prints current metrics at end of epoch.
        Returns loss, mean accuracy and mean per class accuracy for this epoch on the validation data set.

        epoch_idx (int): Current epoch number
        """
        self.model.eval()
        self.val_metric.reset()

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for x, y in tqdm(self.val_loader, desc=f"Val Epoch {epoch_idx}"):
                x, y = x.to(self.device), y.to(self.device)

                out = self.model(x)
                loss = self.loss_fn(out, y)

                self.val_metric.update(out, y)
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        acc = self.val_metric.accuracy()
        pc_acc = self.val_metric.per_class_accuracy()

        print(f"\n[Val] Epoch {epoch_idx}")
        print(f"\n______epoch {epoch_idx}\n")
        print(f"accuracy: {acc:.4f}")
        print(f"per class accuracy: {pc_acc:.4f}")

        for class_idx, classname in enumerate(self.val_metric.classes):
            total = self.val_metric.total_pred[classname]
            correct = self.val_metric.correct_pred[classname]
            if total > 0:
                acc_cls = correct / total
                print(f"Accuracy for class: {classname:<6} is {acc_cls:.2f}")
            else:
                print(f"Accuracy for class: {classname:<6} is N/A")

        return avg_loss, acc, pc_acc


    def train(self) -> None:
        """
        Full training logic that loops over num_epochs and
        uses the _train_epoch and _val_epoch methods.
        Save the model if mean per class accuracy on validation data set is higher
        than currently saved best mean per class accuracy.
        """
        print("Starting training...\n")

        for epoch in range(1, self.num_epochs + 1):
            # === Train one epoch ===
            train_loss, train_acc, train_pcacc = self._train_epoch(epoch)

            # === Log to W&B if active ===
            if self.wandb_logger:
                self.wandb_logger.log({
                    "train/loss": train_loss,
                    "train/accuracy": train_acc,
                    "train/per_class_accuracy": train_pcacc,
                    "epoch": epoch
                })

            # === Validation ===
            if epoch % self.val_frequency == 0 or epoch == self.num_epochs:
                val_loss, val_acc, val_pcacc = self._val_epoch(epoch)

                if self.wandb_logger:
                    self.wandb_logger.log({
                        "val/loss": val_loss,
                        "val/accuracy": val_acc,
                        "val/per_class_accuracy": val_pcacc,
                        "epoch": epoch
                    })

                # Save if best so far
                if val_pcacc > self.best_pcacc:
                    self.best_pcacc = val_pcacc
                    model_path = self.training_save_dir / "best_model.pth"
                    print(f"Saving new best model at epoch {epoch} with per-class accuracy {val_pcacc:.4f}")
                    self.model.save(model_path)

            # === LR Scheduler step ===
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()