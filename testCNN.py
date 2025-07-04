# test_cnn.py
import os
import torch
import torchvision.transforms.v2 as v2
from pathlib import Path
from torch.utils.data import DataLoader

from assignment_1_code.models.cnn import YourCNN
from assignment_1_code.metrics import Accuracy
from assignment_1_code.datasets.cifar10 import CIFAR10Dataset
from assignment_1_code.datasets.dataset import Subset


MODEL_PATH = "saved_models_cnn/best_model.pth"


def test():
    # ─── Transforms
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])

    # ─── Dataset & DataLoader
    data_dir = Path(r"C:\Users\natha\OneDrive\Documents\tu\cifar-10-batches-py")
    test_data = CIFAR10Dataset(
        fdir=str(data_dir),
        subset=Subset.TEST,
        transform=transform
    )
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

    # ─── Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ─── Model CNN
    model = YourCNN(num_classes=len(test_data.classes))
    # Charge le state dict
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # ─── Loss & Metric
    loss_fn = torch.nn.CrossEntropyLoss()
    metric = Accuracy(classes=test_data.classes)

    total_loss = 0.0
    n_batches = 0
    metric.reset()

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)

            total_loss += loss.item()
            n_batches += 1
            metric.update(out, y)

    avg_loss = total_loss / n_batches
    acc = metric.accuracy()
    pc_acc = metric.per_class_accuracy()

    print(f"\n--- Test results (YourCNN) ---")
    print(f"Test loss: {avg_loss:.6f}\n")
    print(f"accuracy: {acc:.4f}")
    print(f"per class accuracy: {pc_acc:.4f}\n")

    for classname in test_data.classes:
        total = metric.total_pred[classname]
        correct = metric.correct_pred[classname]
        if total > 0:
            print(f"Accuracy for class: {classname:<6} is {correct/total:.2f}")
        else:
            print(f"Accuracy for class: {classname:<6} is N/A")


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    test()
