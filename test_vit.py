# test_vit.py
import os
import torch
import torchvision.transforms.v2 as v2
from pathlib import Path
from torch.utils.data import DataLoader

from assignment_1_code.models.vit import ViT
from assignment_1_code.metrics import Accuracy
from assignment_1_code.datasets.cifar10 import CIFAR10Dataset
from assignment_1_code.datasets.dataset import Subset

# Path to your best ViT checkpoint
MODEL_PATH = "saved_models_vit/best_model.pth"


def test():
    # ─── Data transforms for test (no augmentation) ────────────────────────
    transform = v2.Compose([
        v2.ToImage(),                                # convert from array to PIL/Image
        v2.ToDtype(torch.float32, scale=True),       # cast to float tensor in [0,1]
        v2.Normalize(mean=[0.485, 0.456, 0.406],      # normalize with ImageNet stats
                     std=[0.229, 0.224, 0.225]),
    ])

    # ─── Prepare test dataset & loader ────────────────────────────────────
    data_dir = Path(r"C:\Users\natha\OneDrive\Documents\tu\cifar-10-batches-py")
    test_ds = CIFAR10Dataset(
        fdir=str(data_dir),
        subset=Subset.TEST,
        transform=transform
    )
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    # ─── Device setup ─────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ─── Instantiate ViT model and load weights ────────────────────────────
    model = ViT(
        img_size=32,
        patch_size=4,
        emb_size=128,
        num_classes=len(test_ds.classes),
        depth=6,
        heads=8,
        dropout=0.1
    )
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # ─── Loss function & accuracy metric ──────────────────────────────────
    loss_fn = torch.nn.CrossEntropyLoss()
    metric = Accuracy(classes=test_ds.classes)

    # ─── Run test loop ─────────────────────────────────────────────────────
    total_loss = 0.0
    total_batches = 0
    metric.reset()

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item()
            total_batches += 1
            metric.update(outputs, labels)

    avg_loss = total_loss / total_batches
    acc = metric.accuracy()
    pc_acc = metric.per_class_accuracy()

    # ─── Print out results ─────────────────────────────────────────────────
    print(f"\n--- Test results (ViT) ---")
    print(f"Test loss: {avg_loss:.6f}")
    print(f"Overall accuracy: {acc:.4f}")
    print(f"Per-class accuracy: {pc_acc:.4f}\n")

    for classname in test_ds.classes:
        total = metric.total_pred[classname]
        correct = metric.correct_pred[classname]
        acc_cls = correct / total if total > 0 else float('nan')
        print(f"Class {classname:<6}: accuracy = {acc_cls:.2f}")


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    test()
