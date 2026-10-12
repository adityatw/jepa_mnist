"""
 * MIT License
 * * Copyright (c) 2026 Aditya Wresniyandaka
 * * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 
traditional_nn.py
------------------
The FAMILIAR approach: supervised classification.

Idea:  image  --->  neural network  --->  predicted digit (0..9)
We compare the prediction to the TRUE label and adjust the weights.

Key properties to keep in mind (this is what JEPA will contrast against):
  * It NEEDS labels for every training image.
  * The features it learns are shaped entirely by the "which digit is this?"
    task. They may not transfer to other tasks.
  * The output is directly the thing we care about (the class).

Run:
    python traditional_nn.py
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import get_dataloaders


# ---------------------------------------------------------------------------
# The model: a small convolutional neural network (CNN) that ends in a
# 10-way classifier. Conv layers extract visual features; the final Linear
# layer turns features into 10 class scores (one per digit).
# ---------------------------------------------------------------------------
class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Feature extractor: two conv blocks. Each block: conv -> ReLU -> pool.
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # [B,1,28,28] -> [B,32,28,28]
            nn.ReLU(),
            nn.MaxPool2d(2),                              # -> [B,32,14,14]
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # -> [B,64,14,14]
            nn.ReLU(),
            nn.MaxPool2d(2),                              # -> [B,64,7,7]
        )
        # Classifier head: flatten features -> hidden layer -> 10 class scores.
        self.classifier = nn.Sequential(
            nn.Flatten(),                 # [B,64,7,7] -> [B, 64*7*7=3136]
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),  # -> [B, 10] raw scores ("logits")
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


@torch.no_grad()
def evaluate(model, loader, device):
    """Return classification accuracy (fraction correct) on a data loader."""
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)  # pick the highest-scoring class
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


def train(epochs=3, batch_size=256, lr=1e-3, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[traditional] device = {device}")

    train_loader, test_loader = get_dataloaders(batch_size=batch_size)
    model = SmallCNN().to(device)
    # GPU NOTE: `device` above is already "cuda" when a GPU is present, so the
    # .to(device) call already moves the model onto the GPU. To force it
    # explicitly on your GPU laptop you could instead write:
    # model = SmallCNN().to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Track per-epoch metrics so we can plot them afterwards.
    history = {"epoch": [], "train_loss": [], "test_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            # Cross-entropy: the standard classification loss. It rewards the
            # model for putting high score on the correct label. THIS is where
            # the labels enter the picture.
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()      # compute gradients
            optimizer.step()     # update weights
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        acc = evaluate(model, test_loader, device)
        history["epoch"].append(epoch)
        history["train_loss"].append(epoch_loss)
        history["test_acc"].append(acc * 100)
        print(
            f"[traditional] epoch {epoch}/{epochs}  "
            f"train_loss={epoch_loss:.4f}  "
            f"test_acc={acc*100:.2f}%"
        )

    plot_history(history)
    return model


def plot_history(history, path="traditional_nn_metrics.png"):
    """Plot train loss and test accuracy by epoch, side by side, and save."""
    epochs = history["epoch"]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))

    ax_loss.plot(epochs, history["train_loss"], marker="o", color="tab:red")
    ax_loss.set_title("Training loss by epoch")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("cross-entropy loss")
    ax_loss.grid(True, alpha=0.3)

    ax_acc.plot(epochs, history["test_acc"], marker="o", color="tab:blue")
    ax_acc.set_title("Test accuracy by epoch")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy (%)")
    ax_acc.grid(True, alpha=0.3)

    fig.suptitle("Traditional supervised CNN")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[traditional] saved metrics chart to {path}")


if __name__ == "__main__":
    epochs = 50
    train(epochs=epochs, batch_size=256, lr=1e-3)
