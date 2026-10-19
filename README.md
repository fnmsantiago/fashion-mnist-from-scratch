# Fashion-MNIST from scratch

This is a 10-class classifier written from scratch using NumPy and the publicly available Fashion-MNIST dataset. Each implementation written from scratch is accompanied by a set of unit tests to ensure robustness. After which, the project results, visualized via Jupyter notebook, are validated against the outputs of a PyTorch implementation.

## Goal
I wanted to practice the Machine/Deep Learning concepts I learned, that is:
- Forward Propagation
- Backpropagation
- Gradient Checking
- Holdout Validation (Dev) Set
- Mini-batch Processing
- Early Stopping
- Confusion Matrix
- Precision-Recall

Other notable features:
- Unit Tests using pytest
- matplotlib and Jupyter for visualization


## Results

|                        | from scratch (NumPy) | PyTorch  |
|------------------------|----------------------|----------|
| Test accuracy          | 86.54%               | 86.94%   |
| Epochs (early stopped) | 10                   | 11       |
| Training time (CPU)    | 5.99 sec             | 4.32 sec |

Accuracy and epochs are reproducible — every run in both notebooks is seeded.
Training time is wall-clock and varies slightly with machine load.

### Confusion Matrix
![Confusion matrix (row-normalized)](results/confusion_matrix.png)

### Precision/Recall
![Precision/Recall](results/precision_recall.png)

## What the errors reveal
The bar chart reveals that **Shirts** have the **lowest recall**. Looking at the confusion matrix, we see that recall is low because **Shirts** are confused with: **T-Shirts**, **Pullover**, and **Coats**. **Coats** are the most common mistake. A lot of real **Shirts** are predicted as **Coats**.

**Coats** have **high recall**, suggesting that it catches almost all **Coats**. However, it also has **lowest precision** as it misclassifies **Pullovers** and **Shirts** as **Coats**. This aligns with the observations on the previous paragraph.

## Architecture

A fully-connected network with two hidden layers. The class itself is L-layer
(it takes a list of layer sizes), though the notebooks use:

```
784 inputs -> 64 (ReLU) -> 32 (ReLU) -> 10 (softmax)
```

- **Hidden layers**: ReLU, initialized with He scaling (`sqrt(2 / previous layer size)`)
- **Output layer**: softmax over the 10 classes, trained with categorical cross-entropy
- **Training**: mini-batch gradient descent (batch size 64, learning rate 0.1) with early stopping on the validation cost (patience 3)
- **Gradients**: backpropagation written by hand.

### Data convention

Each sample is placed in a column, so inputs
are `(n_x, m)` and predictions/labels are `(n_classes, m)`, where: `m` is the number of samples, `n_x` is the number of features of the inputs, `n_classes` is the number of classes to predict. 

The PyTorch twin expects the opposite layout (samples as rows).

This is the only point where the From-Scratch and PyTorch implementation differ in.


## Correctness
Gradient-checking was employed to ensure the from-scratch implementation of Backpropagation is correct. Aside from that, numerous unit tests were implemented for robustness.


## Repository layout

```
src/          model, activations, losses, metrics, gradient checking
notebooks/    01 from scratch, 02 PyTorch twin (cell-for-cell mirror)
tests/        58 unit tests
results/      metrics JSON from both training runs
```


## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```


## Running it

Run the unit tests from the project root:

```bash
pytest
```

Then open the notebooks in order, selecting the venv as the kernel:

1. `notebooks/01_train_from_scratch.ipynb` — trains the NumPy implementation, evaluates it, and writes `results/from_scratch.json`
2. `notebooks/02_pytorch_comparison.ipynb` — trains the equivalent PyTorch model and compares it against that same JSON

The first run downloads Fashion-MNIST through scikit-learn (about a minute); after that it is cached locally.


