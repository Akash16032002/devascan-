import os
import json
from typing import List, Tuple
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

import numpy as np
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix
)

from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfgen import canvas


# --------- CONFIG ---------
DATASET_ROOT = os.path.join(
    "..",
    "dataset",
    "devanagari+handwritten+character+dataset",
    "DevanagariHandwrittenCharacterDataset",
    "train",
)

BATCH_SIZE = 128
NUM_EPOCHS = 5
LR = 1e-3

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "ocr_cnn.pth")
LABELS_PATH = os.path.join(MODEL_DIR, "ocr_labels.json")
METRICS_PDF_PATH = os.path.join(MODEL_DIR, "model_evaluation_report.pdf")
CM_PDF_PATH = os.path.join(MODEL_DIR, "confusion_matrix.pdf")


# --------- DATASET ---------
class DevanagariCharDataset(Dataset):
    def __init__(self, root: str, transform=None):
        self.root = root
        self.transform = transform

        self.class_names: List[str] = sorted(
            d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
        )
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        self.samples: List[Tuple[str, int]] = []
        for class_name in self.class_names:
            class_dir = os.path.join(root, class_name)
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    self.samples.append(
                        (os.path.join(class_dir, fname), self.class_to_idx[class_name])
                    )

        print(f"Found {len(self.samples)} images across {len(self.class_names)} classes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("L")

        if self.transform:
            img = self.transform(img)

        return img, label


# --------- MODEL ---------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = DevanagariCharDataset(DATASET_ROOT, transform=transform)

    if len(dataset) == 0:
        print("[ERROR] Dataset is empty.")
        return

    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = SimpleCNN(len(dataset.class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # --------- TRAINING ---------
    for epoch in range(NUM_EPOCHS):
        model.train()
        correct, total, loss_sum = 0, 0, 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} "
              f"Train Loss: {loss_sum/total:.4f}, "
              f"Train Acc: {correct/total:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset.class_names, f, indent=2, ensure_ascii=False)

    print("Model and labels saved.")

    # --------- EVALUATION ---------
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    accuracy = accuracy_score(all_labels, all_preds)
    precision_macro = precision_score(all_labels, all_preds, average="macro")
    recall_macro = recall_score(all_labels, all_preds, average="macro")
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    precision_micro = precision_score(all_labels, all_preds, average="micro")
    recall_micro = recall_score(all_labels, all_preds, average="micro")
    f1_micro = f1_score(all_labels, all_preds, average="micro")

    y_true = np.eye(len(dataset.class_names))[all_labels]
    auc_pr = average_precision_score(y_true, all_probs, average="macro")

    # --------- METRICS PDF ---------
    c = canvas.Canvas(METRICS_PDF_PATH, pagesize=A4)
    y = 800

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Model Evaluation Report")
    y -= 30

    c.setFont("Helvetica", 11)
    for line in [
        f"Accuracy               : {accuracy:.4f}",
        f"Precision (Macro)      : {precision_macro:.4f}",
        f"Recall (Macro)         : {recall_macro:.4f}",
        f"F1 Score (Macro)       : {f1_macro:.4f}",
        f"Precision (Micro)      : {precision_micro:.4f}",
        f"Recall (Micro)         : {recall_micro:.4f}",
        f"F1 Score (Micro)       : {f1_micro:.4f}",
        f"AUC-PR (Macro)         : {auc_pr:.4f}",
    ]:
        c.drawString(60, y, line)
        y -= 20

    c.drawString(60, y, "Cross-Entropy Loss used for training.")
    c.showPage()
    c.save()

    # --------- CONFUSION MATRIX PDF ---------
    cm = confusion_matrix(all_labels, all_preds)
    max_classes = min(15, cm.shape[0])
    cm = cm[:max_classes, :max_classes]

    table_data = [["T\\P"] + list(range(max_classes))]
    for i in range(max_classes):
        table_data.append([i] + list(cm[i]))

    doc = SimpleDocTemplate(CM_PDF_PATH, pagesize=A4)
    table = Table(table_data)
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
    ]))

    doc.build([table])

    print("Evaluation PDFs generated successfully.")


if __name__ == "__main__":
    main()
