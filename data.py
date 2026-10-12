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
 
data.py
-------
Shared MNIST data loading used by both the traditional supervised model
(traditional_nn.py) and the self-supervised JEPA model (jepa_mnist.py).

MNIST is 70,000 grayscale images of handwritten digits (0-9), each 28x28
pixels. 60,000 are for training, 10,000 for testing.

Nothing here is JEPA-specific yet. This is just "how we get the pixels".
"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# MNIST's dataset-wide pixel mean and standard deviation (well-known constants).
# We standardize inputs (subtract mean, divide by std) so training is stable.
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def get_transform():
    """Turn a PIL image into a normalized float tensor of shape [1, 28, 28]."""
    return transforms.Compose(
        [
            transforms.ToTensor(),  # -> float tensor in [0, 1], shape [1, 28, 28]
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),  # standardize
        ]
    )


def get_dataloaders(batch_size=256, data_root="./data", num_workers=0):
    """
    Download (once) and return train/test DataLoaders.

    A DataLoader hands us the dataset in shuffled mini-batches. Each item it
    yields is a tuple (images, labels):
      images: tensor of shape [batch_size, 1, 28, 28]
      labels: tensor of shape [batch_size] with integers 0..9

    NOTE: JEPA will ignore the labels during pre-training. We only keep them
    around so we can (a) train the traditional network and (b) later measure
    how good JEPA's features are with a small "linear probe".
    """
    transform = get_transform()

    train_ds = datasets.MNIST(
        root=data_root, train=True, download=True, transform=transform
    )
    test_ds = datasets.MNIST(
        root=data_root, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,  # keeps every batch the same size (simpler for JEPA)
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, test_loader


if __name__ == "__main__":
    # Quick sanity check: run `python data.py` to confirm data downloads and
    # the shapes are what we expect.
    train_loader, test_loader = get_dataloaders(batch_size=8)
    images, labels = next(iter(train_loader))
    print("images batch shape:", tuple(images.shape))  # (8, 1, 28, 28)
    print("labels batch shape:", tuple(labels.shape))  # (8,)
    print("labels example:", labels.tolist())
    print("pixel min/max after normalize:", float(images.min()), float(images.max()))
