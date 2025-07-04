import argparse
import os
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

import torchvision.transforms.v2 as v2

from assignment_1_code.models.vit import ViT
from assignment_1_code.metrics import Accuracy
from assignment_1_code.trainer import ImgClassificationTrainer
from assignment_1_code.datasets.cifar10 import CIFAR10Dataset
from assignment_1_code.datasets.dataset import Subset


def train(args):
    # ---- Transforms ----
    cifar_mean = [0.4914, 0.4822, 0.4465]
    cifar_std  = [0.2470, 0.2435, 0.2616]

    train_transform = v2.Compose([
        v2.ToImage(),
        v2.RandomCrop(size=(32, 32), padding=4),
        v2.RandomHorizontalFlip(),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=cifar_mean, std=cifar_std),
    ])
    val_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=cifar_mean, std=cifar_std),
    ])

    data_dir = Path(r"C:\Users\natha\OneDrive\Documents\tu\cifar-10-batches-py")
    train_data = CIFAR10Dataset(
        fdir=str(data_dir), subset=Subset.TRAINING, transform=train_transform
    )
    val_data = CIFAR10Dataset(
        fdir=str(data_dir), subset=Subset.VALIDATION, transform=val_transform
    )

    # ---- Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Model
    model = ViT(
        img_size=32, patch_size=4, emb_size=128,
        num_classes=len(train_data.classes),
        depth=6, heads=8, dropout=0.2
    ).to(device)

    # ---- Optimizer + Scheduler + Loss ----
    optimizer = AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.05
    )

    # Scheduler : warm-up linars 500 iters, then CosineAnnealing
    total_steps = (len(train_data) // args.batch_size) * args.num_epochs
    def lr_lambda(step):
        if step < 500:
            return float(step) / 500
        return 0.5 * (1 + math.cos(math.pi * (step - 500) / (total_steps - 500)))

    scheduler = LambdaLR(optimizer, lr_lambda)

    loss_fn = nn.CrossEntropyLoss()

    # ---- Metrics & Trainer ----
    train_metric = Accuracy(classes=train_data.classes)
    val_metric   = Accuracy(classes=val_data.classes)
    val_frequency = 1  # validation chaque epoch

    save_dir = Path("saved_models_vit")
    save_dir.mkdir(exist_ok=True)

    trainer = ImgClassificationTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        lr_scheduler=scheduler,
        train_metric=train_metric,
        val_metric=val_metric,
        train_data=train_data,
        val_data=val_data,
        device=device,
        num_epochs=args.num_epochs,
        training_save_dir=save_dir,
        batch_size=args.batch_size,
        val_frequency=val_frequency,
    )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Vision Transformer on CIFAR-10"
    )
    parser.add_argument(
        "-d", "--gpu_id", default="0", type=str,
        help="index of which GPU to use"
    )
    parser.add_argument(
        "-e", "--num_epochs", default=130, type=int,
        help="number of training epochs"
    )
    parser.add_argument(
        "-b", "--batch_size", default=128, type=int,
        help="batch size for training"
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    train(args)
