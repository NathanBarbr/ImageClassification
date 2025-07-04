import pickle
from typing import Tuple
import numpy as np
import os


from assignment_1_code.datasets.dataset import Subset, ClassificationDataset


class CIFAR10Dataset(ClassificationDataset):
    """
    Custom CIFAR-10 Dataset.
    """

    def __init__(self, fdir: str, subset: Subset, transform=None):
        """
        Initializes the CIFAR-10 dataset.
        """
        self.classes = (
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )

        self.fdir = fdir
        self.subset = subset
        self.transform = transform

        self.images, self.labels = self.load_cifar()

    def load_cifar(self) -> Tuple:
        """
        Loads the dataset from a directory fdir that contains the Python version
        of the CIFAR-10, i.e. files "data_batch_1", "test_batch" and so on.
        Raises ValueError if fdir is not a directory or if a file inside it is missing.

        The subsets are defined as follows:
          - The training set contains all images from "data_batch_1" to "data_batch_4", in this order.
          - The validation set contains all images from "data_batch_5".
          - The test set contains all images from "test_batch".

        Depending on which subset is selected, the corresponding images and labels are returned.

        Images are loaded in the order they appear in the data files
        and returned as uint8 numpy arrays with shape (32, 32, 3), in RGB channel order.
        """

        # 1. Check that fdir is a directory
        if not os.path.isdir(self.fdir):
            raise ValueError(f"Directory not found: {self.fdir}")

        # 2. Determine which files to load
        if self.subset == Subset.TRAINING:
            batch_files = [f"data_batch_{i}" for i in range(1, 5)]
        elif self.subset == Subset.VALIDATION:
            batch_files = ["data_batch_5"]
        elif self.subset == Subset.TEST:
            batch_files = ["test_batch"]
        else:
            raise ValueError(f"Unknown subset: {self.subset}")

        data_list = []
        label_list = []

        # 3. Load each batch
        for fname in batch_files:
            path = os.path.join(self.fdir, fname)
            if not os.path.isfile(path):
                raise ValueError(f"Missing CIFAR file: {path}")

            with open(path, 'rb') as f:
                batch = pickle.load(f, encoding='latin1')

            # Expect keys 'data' and 'labels'
            data = batch.get('data')
            labels = batch.get('labels') or batch.get('fine_labels')
            if data is None or labels is None:
                raise ValueError(f"Unexpected file format in {path}")

            data_list.append(data)
            label_list.extend(labels)

        # 4. Concatenate all batches
        all_data = np.vstack(data_list)  # shape (N, 3072)
        all_labels = np.array(label_list, dtype=np.int64)

        # 5. Reshape to (N, 3, 32, 32) then transpose to (N, 32, 32, 3)
        N = all_data.shape[0]
        images = all_data.reshape(N, 3, 32, 32)
        images = images.transpose(0, 2, 3, 1).astype(np.uint8)

        return images, all_labels

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self.images)


    def __getitem__(self, idx: int) -> Tuple:
        """
        Returns the idx-th sample in the dataset, which is a tuple,
        consisting of the image and labels.
        Applies transforms if not None.
        Raises IndexError if the index is out of bounds.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} is out of bounds.")

        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

    def num_classes(self) -> int:
        """
        Returns the number of classes.
        """
        return len(self.classes)
