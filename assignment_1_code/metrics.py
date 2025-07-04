from abc import ABCMeta, abstractmethod
import torch


class PerformanceMeasure(metaclass=ABCMeta):
    """
    A performance measure.
    """

    @abstractmethod
    def reset(self):
        """
        Resets internal state.
        """

        pass

    @abstractmethod
    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        """
        Update the measure by comparing predicted data with ground-truth target data.
        Raises ValueError if the data shape or values are unsupported.
        """

        pass

    @abstractmethod
    def __str__(self) -> str:
        """
        Return a string representation of the performance.
        """

        pass


class Accuracy(PerformanceMeasure):
    """
    Average classification accuracy.
    """

    def __init__(self, classes) -> None:
        self.classes = classes
        self.reset()

    def reset(self) -> None:
        """
        Resets the internal state.
        """
        self.correct_pred = {classname: 0 for classname in self.classes}
        self.total_pred = {classname: 0 for classname in self.classes}
        self.n_matching = 0  # number of correct predictions
        self.n_total = 0

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update the measure by comparing predicted data with ground-truth target data.
        prediction must have shape (batchsize,n_classes) with each row being a class-score vector.
        target must have shape (batchsize,) and values between 0 and c-1 (true class labels).
        Raises ValueError if the data shape or values are unsupported.
        [len(prediction.shape) should be equal to 2, and len(target.shape) should be equal to 1.]
        """

        # Type checks
        if not isinstance(prediction, torch.Tensor) or not isinstance(target, torch.Tensor):
            raise ValueError("prediction and target must be torch.Tensor")

        # Shape checks
        if prediction.ndim != 2:
            raise ValueError(f"prediction must be 2D (batch, classes), got shape {prediction.shape}")
        if target.ndim != 1:
            raise ValueError(f"target must be 1D (batch,), got shape {target.shape}")
        if prediction.shape[0] != target.shape[0]:
            raise ValueError("batch size of prediction and target must match")

        batch_size, n_classes = prediction.shape

        # Compute predicted labels
        pred_labels = torch.argmax(prediction, dim=1)
        if pred_labels.max().item() >= n_classes or target.max().item() >= n_classes:
            raise ValueError("class indices in prediction or target exceed number of classes")

        # Overall counts
        matches = (pred_labels == target)
        self.n_matching += matches.sum().item()
        self.n_total += batch_size

        # Per-class counts
        for class_idx, classname in enumerate(self.classes):
            mask_class = (target == class_idx)
            n_in_class = mask_class.sum().item()
            if n_in_class > 0:
                # correct in this class
                correct_in_class = (matches & mask_class).sum().item()
                self.correct_pred[classname] += correct_in_class
                self.total_pred[classname] += n_in_class

    def __str__(self):
        """
        Return a string representation of the performance,
        showing overall accuracy and per-class accuracy.
        """
        overall = self.accuracy()
        mean_cls = self.per_class_accuracy()
        return f"Accuracy: {overall:.4f}, Mean Class Accuracy: {mean_cls:.4f}"

    def accuracy(self) -> float:
        """
        Compute and return the accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        """
        if self.n_total == 0:
            return 0.0
        return self.n_matching / self.n_total


    def per_class_accuracy(self) -> float:
        """
        Compute and return the per class accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        """
        if self.n_total == 0:
            return 0.0

        accs = []
        for classname in self.classes:
            total = self.total_pred[classname]
            if total > 0:
                accs.append(self.correct_pred[classname] / total)
            else:
                accs.append(0.0)
        return sum(accs) / len(accs)