import argparse
import os
import torch
import torchvision.transforms.v2 as v2
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn as nn


from assignment_1_code.models.cnn import YourCNN
from assignment_1_code.metrics import Accuracy
from assignment_1_code.trainer import ImgClassificationTrainer
from assignment_1_code.datasets.cifar10 import CIFAR10Dataset
from assignment_1_code.datasets.dataset import Subset


def train(args):
    # Transforms
    train_transform = v2.Compose([
        v2.ToImage(),
        v2.RandomHorizontalFlip(),
        v2.RandomResizedCrop(size=(32, 32), scale=(0.8, 1.0)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])
    val_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])

    # Datasets
    data_dir = Path(r"C:\Users\natha\OneDrive\Documents\tu\cifar-10-batches-py")
    train_data = CIFAR10Dataset(
        fdir=str(data_dir),
        subset=Subset.TRAINING,
        transform=train_transform
    )
    val_data = CIFAR10Dataset(
        fdir=str(data_dir),
        subset=Subset.VALIDATION,
        transform=val_transform
    )

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = YourCNN(num_classes=len(train_data.classes)).to(device)

    # Optimizer, Scheduler, Loss

    optimizer = AdamW(model.parameters(),lr=0.001,
                      amsgrad=True,weight_decay=1e-4)
    lr_scheduler = ExponentialLR(optimizer, gamma=0.9)

    loss_fn = nn.CrossEntropyLoss()

    # Metrics
    train_metric = Accuracy(classes=train_data.classes)
    val_metric = Accuracy(classes=val_data.classes)
    val_frequency = 5


    model_save_dir = Path("saved_models_cnn")
    model_save_dir.mkdir(exist_ok=True)

    # Trainer
    trainer = ImgClassificationTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        lr_scheduler=lr_scheduler,
        train_metric=train_metric,
        val_metric=val_metric,
        train_data=train_data,
        val_data=val_data,
        device=device,
        num_epochs=args.num_epochs,
        training_save_dir=model_save_dir,
        batch_size=128,
        val_frequency=val_frequency,
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training YourCNN on CIFAR-10")
    parser.add_argument(
        "-d", "--gpu_id", default="0", type=str,
        help="index of which GPU to use"
    )
    parser.add_argument(
        "-e", "--num_epochs", default=30, type=int,
        help="number of training epochs"
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    train(args)
