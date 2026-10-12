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
 
jepa_mnist.py
-------------
A minimal JEPA (Joint-Embedding Predictive Architecture) on MNIST.

BIG PICTURE
===========
The traditional network (traditional_nn.py) learns:   image -> label
JEPA instead learns (WITHOUT any labels):

    partial view of a digit  ->  (in embedding space)  ->  representation of
                                                           the fuller view

It never reconstructs pixels and never sees a class label during
pre-training. It just learns to "fill in the missing concept" in an abstract
embedding space. Afterwards we check whether those learned features are
useful by training a tiny linear classifier on top of them (a "linear probe").

THE THREE PARTS OF A JEPA
=========================
  1. context encoder (online)  : encodes a MASKED image (the "context")
  2. target encoder  (EMA)     : encodes the FULL image (the "target");
                                 it is a slow-moving copy of the context
                                 encoder and we NEVER backprop through it.
  3. predictor                 : maps the context embedding to a PREDICTION
                                 of the target embedding.

Loss = distance between predictor(context) and stop_grad(target),
measured in embedding space.

WHY IT DOESN'T "CHEAT"
======================
A lazy solution would map every image to the same constant vector, making
the loss zero (this is called "representation collapse"). Two design choices
prevent it:
  * stop-gradient on the target (we don't let the target move to meet the
    prediction), and
  * the target encoder is an exponential moving average (EMA) of the online
    encoder, so it changes slowly.
This asymmetry (predictor + EMA target) is the same recipe used by BYOL and
by Meta's I-JEPA.

HOW THIS RELATES TO THE "REAL" I-JEPA
=====================================
Full I-JEPA uses a Vision Transformer and predicts the embeddings of several
masked patch-blocks using their positions. Here we keep it to one masked
block + a CNN so the core idea is easy to follow. The learning principle is
identical.

Run:
    python jepa_mnist.py
"""

import copy

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import get_dataloaders


# ---------------------------------------------------------------------------
# 1. The encoder: image -> embedding vector (no classifier here!).
#    Both the context encoder and the target encoder are instances of this.
# ---------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # [B,1,28,28]->[B,32,28,28]
            nn.ReLU(),
            nn.MaxPool2d(2),                               # -> [B,32,14,14]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # -> [B,64,14,14]
            nn.ReLU(),
            nn.MaxPool2d(2),                               # -> [B,64,7,7]
        )
        self.project = nn.Sequential(
            nn.Flatten(),                  # -> [B, 3136]
            nn.Linear(64 * 7 * 7, embed_dim),  # -> [B, embed_dim]
        )

    def forward(self, x):
        return self.project(self.features(x))


# ---------------------------------------------------------------------------
# 2. The predictor: context embedding -> predicted target embedding.
#    A small MLP. Keeping it a separate, asymmetric module is important for
#    avoiding collapse.
# ---------------------------------------------------------------------------
class Predictor(nn.Module):
    def __init__(self, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Masking: build the "context" view by hiding a random square block of the
# image. The unmasked original image is the "target" view.
#
#   target view :  the full digit
#   context view:  the same digit with a square erased -> the model must
#                  infer the whole concept from the partial view.
# ---------------------------------------------------------------------------
def make_context_view(images, block=12):
    """Return a copy of `images` with a random `block`x`block` square zeroed."""
    B, C, H, W = images.shape
    ctx = images.clone()
    top = torch.randint(0, H - block + 1, (B,))
    left = torch.randint(0, W - block + 1, (B,))
    for i in range(B):
        ctx[i, :, top[i] : top[i] + block, left[i] : left[i] + block] = 0.0
    return ctx


# ---------------------------------------------------------------------------
# JEPA loss: cosine distance in embedding space (BYOL-style).
# We L2-normalize both vectors, so this equals 2 - 2*cosine_similarity.
# Minimizing it means "make the predicted embedding point the same direction
# as the target embedding".
# ---------------------------------------------------------------------------
def jepa_loss(pred, target):
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    return 2 - 2 * (pred * target).sum(dim=-1).mean()


@torch.no_grad()
def ema_update(target_encoder, online_encoder, momentum=0.996):
    """
    Move the target encoder a tiny step toward the online encoder:
        target = momentum * target + (1 - momentum) * online
    High momentum (0.996) => target changes slowly => stable targets.
    We also copy BatchNorm/running buffers so the two stay consistent.
    """
    for tp, op in zip(target_encoder.parameters(), online_encoder.parameters()):
        tp.data.mul_(momentum).add_(op.data, alpha=1 - momentum)
    for tb, ob in zip(target_encoder.buffers(), online_encoder.buffers()):
        tb.data.copy_(ob.data)


def pretrain_jepa(epochs=5, batch_size=256, lr=1e-3, embed_dim=128,
                  block=12, momentum=0.996, device=None):
    """Self-supervised pre-training. Returns the trained target encoder."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[jepa] device = {device}")

    train_loader, _ = get_dataloaders(batch_size=batch_size)

    online_encoder = Encoder(embed_dim).to(device)
    predictor = Predictor(embed_dim).to(device)
    # GPU NOTE: `device` above is already "cuda" when a GPU is present, so the
    # .to(device) calls above already put these models on the GPU. To force it
    # explicitly on your GPU laptop you could instead write:
    # online_encoder = Encoder(embed_dim).to("cuda")
    # predictor = Predictor(embed_dim).to("cuda")

    # The target encoder starts as an exact copy of the online encoder, but
    # its parameters do NOT receive gradients (updated only via EMA).
    target_encoder = copy.deepcopy(online_encoder).to(device)
    for p in target_encoder.parameters():
        p.requires_grad = False

    # Only the online encoder and predictor are optimized by gradient descent.
    params = list(online_encoder.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    # Track per-epoch pre-training loss so we can plot it afterwards.
    history = {"epoch": [], "pretrain_loss": []}

    for epoch in range(1, epochs + 1):
        online_encoder.train()
        predictor.train()
        running_loss = 0.0

        for images, _ in train_loader:  # NOTE: labels are ignored (the "_")
            images = images.to(device)

            context_view = make_context_view(images, block=block)  # masked
            target_view = images                                    # full

            # Online path (gets gradients): encode context, then predict.
            z_context = online_encoder(context_view)
            z_pred = predictor(z_context)

            # Target path (NO gradients): encode the full image.
            with torch.no_grad():
                z_target = target_encoder(target_view)

            loss = jepa_loss(z_pred, z_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Slowly pull the target encoder toward the online encoder.
            ema_update(target_encoder, online_encoder, momentum=momentum)

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        history["epoch"].append(epoch)
        history["pretrain_loss"].append(epoch_loss)
        print(
            f"[jepa] epoch {epoch}/{epochs}  "
            f"pretrain_loss={epoch_loss:.4f}"
        )

    return target_encoder, history


# ---------------------------------------------------------------------------
# Linear probe: the standard way to judge self-supervised features.
# We FREEZE the encoder and train ONLY a single Linear(embed_dim -> 10) layer
# on top, using labels. If a mere linear layer on frozen features gets high
# accuracy, the features must already separate the digits well.
# ---------------------------------------------------------------------------
def linear_probe(encoder, epochs=5, batch_size=256, lr=1e-3,
                 embed_dim=128, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_dataloaders(batch_size=batch_size)

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    classifier = nn.Linear(embed_dim, 10).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)

    def accuracy():
        classifier.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                feats = encoder(images)               # frozen features
                preds = classifier(feats).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return correct / total

    # Track per-epoch probe loss and accuracy so we can plot them afterwards.
    history = {"epoch": [], "train_loss": [], "test_acc": []}

    for epoch in range(1, epochs + 1):
        classifier.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                feats = encoder(images)  # encoder is frozen; no grad needed
            logits = classifier(feats)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        acc = accuracy()
        history["epoch"].append(epoch)
        history["train_loss"].append(epoch_loss)
        history["test_acc"].append(acc * 100)
        print(f"[probe] epoch {epoch}/{epochs}  test_acc={acc*100:.2f}%")

    return accuracy(), history


# ---------------------------------------------------------------------------
# Plotting: visualize how the two phases progressed by epoch.
#   * JEPA pre-training loss (self-supervised, no labels)
#   * Linear probe loss and test accuracy (supervised, on frozen features)
# ---------------------------------------------------------------------------
def plot_history(pretrain_hist, probe_hist, path="jepa_mnist_metrics.png"):
    fig, (ax_pre, ax_loss, ax_acc) = plt.subplots(1, 3, figsize=(15, 4))

    ax_pre.plot(
        pretrain_hist["epoch"], pretrain_hist["pretrain_loss"],
        marker="o", color="tab:green",
    )
    ax_pre.set_title("JEPA pre-training loss")
    ax_pre.set_xlabel("epoch")
    ax_pre.set_ylabel("cosine distance loss")
    ax_pre.grid(True, alpha=0.3)

    ax_loss.plot(
        probe_hist["epoch"], probe_hist["train_loss"],
        marker="o", color="tab:red",
    )
    ax_loss.set_title("Linear probe train loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("cross-entropy loss")
    ax_loss.grid(True, alpha=0.3)

    ax_acc.plot(
        probe_hist["epoch"], probe_hist["test_acc"],
        marker="o", color="tab:blue",
    )
    ax_acc.set_title("Linear probe test accuracy")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy (%)")
    ax_acc.grid(True, alpha=0.3)

    fig.suptitle("JEPA: self-supervised pre-training + linear probe")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[jepa] saved metrics chart to {path}")


if __name__ == "__main__":
    EMBED_DIM = 128

    print("=" * 64)
    print("STEP 1: Self-supervised JEPA pre-training (NO labels used)")
    print("=" * 64)
    epochs = 20
    encoder, pretrain_hist = pretrain_jepa(epochs=epochs, embed_dim=EMBED_DIM)

    # Save the trained encoder so visualize.py can load it instantly instead
    # of retraining. We store the weights (state_dict) plus the embed_dim.
    torch.save(
        {"state_dict": encoder.state_dict(), "embed_dim": EMBED_DIM},
        "jepa_encoder.pt",
    )
    print("[jepa] saved trained encoder to jepa_encoder.pt")

    print()
    print("=" * 64)
    print("STEP 2: Linear probe (train ONLY a linear layer on frozen features)")
    print("=" * 64)
    acc, probe_hist = linear_probe(encoder, epochs=epochs, embed_dim=EMBED_DIM)

    # Chart both phases (pre-training loss + probe loss/accuracy) by epoch.
    plot_history(pretrain_hist, probe_hist)

    print()
    print(f"JEPA features + linear probe test accuracy: {acc*100:.2f}%")
    print("Compare this against traditional_nn.py, which trains end-to-end")
    print("with labels the whole time.")
