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
 
visualize.py
------------
Two pictures that make JEPA "click":

  1. viz_masking.png
     Top row  = original digits (the TARGET view the target encoder sees).
     Bottom   = the same digits with a random square erased (the CONTEXT
                view the online encoder sees). JEPA must predict the target's
                embedding from this partial view.

  2. viz_embeddings_pca.png / viz_embeddings_tsne.png
     Each dot is one test image, placed by its JEPA embedding squashed down
     to 2D, and colored by its TRUE digit label. The labels are used ONLY for
     coloring the plot -- never for training. If digits of the same color
     clump together, it means JEPA learned to separate the classes WITHOUT
     ever being told what they are.

HOW IT GETS AN ENCODER
======================
If you have run `python jepa_mnist.py` (which saves jepa_encoder.pt), this
script loads that trained encoder instantly. If the file is missing, it does
a short fallback pre-training so the script still works on its own.

Run:
    python visualize.py
"""

import os

import matplotlib

matplotlib.use("Agg")  # save figures to files instead of opening a window
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from data import MNIST_MEAN, MNIST_STD, get_dataloaders
from jepa_mnist import Encoder, make_context_view, pretrain_jepa

CHECKPOINT = "jepa_encoder.pt"


def get_device():
    # GPU NOTE: this auto-selects "cuda" when a GPU is available, so on your
    # GPU laptop everything below runs on the GPU automatically.
    # To force it explicitly you could uncomment:
    # return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_encoder(device):
    """Load the saved JEPA encoder, or pre-train a quick one if none exists."""
    if os.path.exists(CHECKPOINT):
        ckpt = torch.load(CHECKPOINT, map_location=device)
        embed_dim = ckpt["embed_dim"]
        encoder = Encoder(embed_dim).to(device)
        encoder.load_state_dict(ckpt["state_dict"])
        print(f"[viz] loaded trained encoder from {CHECKPOINT} (dim={embed_dim})")
    else:
        print(f"[viz] {CHECKPOINT} not found -> doing a short fallback pretrain.")
        print("[viz] (run `python jepa_mnist.py` first for better features)")
        embed_dim = 128
        encoder = pretrain_jepa(epochs=1, embed_dim=embed_dim, device=device)
    encoder.eval()
    return encoder


def unnormalize(img):
    """Undo the MNIST normalization so pixels are back in [0, 1] for display."""
    return (img * MNIST_STD + MNIST_MEAN).clamp(0, 1)


def plot_masking(device, n=8, block=12):
    """Save a grid comparing original (target) vs masked (context) views."""
    train_loader, _ = get_dataloaders(batch_size=n)
    images, _ = next(iter(train_loader))
    context = make_context_view(images, block=block)

    fig, axes = plt.subplots(2, n, figsize=(1.4 * n, 3.2))
    for i in range(n):
        axes[0, i].imshow(unnormalize(images[i, 0]).cpu(), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(unnormalize(context[i, 0]).cpu(), cmap="gray")
        axes[1, i].axis("off")
    # Row titles (we use fig.text because the axes have their frames turned off).
    fig.text(0.5, 0.95, "TARGET view (full digit)", ha="center", fontsize=12)
    fig.text(0.5, 0.48, "CONTEXT view (random block erased)", ha="center", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig("viz_masking.png", dpi=120)
    plt.close(fig)
    print("[viz] wrote viz_masking.png")


@torch.no_grad()
def collect_embeddings(encoder, device, n=2000):
    """Run test images through the (frozen) encoder and gather embeddings."""
    _, test_loader = get_dataloaders(batch_size=256)
    feats, labels = [], []
    got = 0
    for images, y in test_loader:
        images = images.to(device)
        feats.append(encoder(images).cpu().numpy())
        labels.append(y.numpy())
        got += images.size(0)
        if got >= n:
            break
    feats = np.concatenate(feats)[:n]
    labels = np.concatenate(labels)[:n]
    return feats, labels


def scatter_2d(coords, labels, title, out_path):
    """Scatter 2D points colored by digit label, with a 0-9 colorbar."""
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(
        coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=8, alpha=0.7
    )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(sc, ax=ax, ticks=range(10))
    cbar.set_label("true digit (used only for coloring, NOT training)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[viz] wrote {out_path}")


def plot_embeddings(encoder, device, n=2000):
    feats, labels = collect_embeddings(encoder, device, n=n)

    # PCA: fast linear projection to 2D.
    pca_2d = PCA(n_components=2).fit_transform(feats)
    scatter_2d(pca_2d, labels, "JEPA embeddings (PCA to 2D)", "viz_embeddings_pca.png")

    # t-SNE: slower, nonlinear, usually shows tighter clusters.
    tsne_2d = TSNE(n_components=2, init="pca", perplexity=30, random_state=0)
    tsne_2d = tsne_2d.fit_transform(feats)
    scatter_2d(tsne_2d, labels, "JEPA embeddings (t-SNE to 2D)", "viz_embeddings_tsne.png")


if __name__ == "__main__":
    device = get_device()
    print(f"[viz] device = {device}")

    # Picture 1: what masking looks like (doesn't need a trained encoder).
    plot_masking(device)

    # Picture 2: how the learned features arrange the digits.
    encoder = load_encoder(device)
    plot_embeddings(encoder, device)

    print("\nDone. Open these PNGs to see the results:")
    print("  viz_masking.png          - target vs context (masked) views")
    print("  viz_embeddings_pca.png   - embeddings in 2D via PCA")
    print("  viz_embeddings_tsne.png  - embeddings in 2D via t-SNE")
