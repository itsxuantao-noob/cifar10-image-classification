# CIFAR-10 Image Classification in PyTorch

Course project for **UW CSE 416**. Five neural network architectures are implemented from scratch in PyTorch and compared on [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), from a linear baseline through fully connected nets to regularized CNNs.

The goal is to measure how architecture, depth, convolution, and regularization change classification accuracy.

## Dataset

CIFAR-10 has 60,000 RGB images of size 32×32 across 10 classes:

Airplane · Automobile · Bird · Cat · Deer · Dog · Frog · Horse · Ship · Truck

Images are normalized to `[-1, 1]`. The official 50,000 / 10,000 train–test split is used as train / validation.

## Models

| Model | Type | Architecture | Best val. acc. |
| --- | --- | --- | --- |
| Baseline | Linear | Flatten → FC(10) | — |
| **A** | Shallow MLP | Flatten → FC(300) → ReLU → FC(10) | 52.37% |
| **B** | Deeper MLP | Flatten → FC(100) → ReLU → FC(60) → ReLU → FC(10) | 52.65% |
| **C** | Simple CNN | Conv(25, 5×5) → ReLU → MaxPool → FC(10) | 65.07% |
| **D** | Deeper CNN | 3 conv blocks (64 / 128 / 256) → FC(256) → FC(10) | 72.00% |
| **E** | Regularized CNN | Same as D, plus BatchNorm after each conv and Dropout(0.4) | **72.59%** |

## Setup

Python 3 with the following packages:

```bash
pip install torch torchvision matplotlib seaborn
```

A GPU is recommended. Training all five models for 10 epochs is slow on CPU. The notebook was originally run in Google Colab with a GPU accelerator.

## Run

Open `cifar10-image-classification-pytorch.ipynb` and run all cells.

Training defaults:

- Optimizer: Adam
- Learning rate: `1e-3`
- Epochs: 10
- Batch size: 100
- Loss: cross-entropy

Set `SAMPLE_DATA = True` near the top of the notebook to train on a small subset if you only want a quick smoke test.

CIFAR-10 downloads automatically into `./data` on the first run.

## Findings

Fully connected models plateau around 52% validation accuracy. They flatten the image and lose spatial structure, so extra depth (A → B) barely helps.

A single convolutional layer (C) jumps accuracy to about 65%. A deeper CNN (D) reaches 72%, but training accuracy climbs to ~92% while validation stalls — a clear overfitting gap.

Batch normalization and dropout in E close that gap a little and produce the best result at **72.59%** validation accuracy.

Takeaway: for this dataset, convolution matters more than width or depth in an MLP, and regularization helps once the CNN is large enough to overfit.

## Notebook outline

1. Data loading and shared `train` / `accuracy` / plotting helpers
2. Model definitions (baseline + A–E)
3. Train and plot each model
4. Side-by-side comparison and write-up
# cifar10-image-classification
