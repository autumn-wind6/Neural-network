# Project 2 README

This repository contains the code implementation for Project 2 of Neural
Network and Deep Learning. The work is organized to match the PDF requirements
as closely as possible.

## PDF Requirements Covered

### 1. Train a Network on CIFAR-10 (60%)

Implemented a custom CIFAR-10 classification network named `ResidualCifarNet`.

The network contains all required basic components:

- Fully-connected layer
- 2D convolutional layer
- 2D pooling layer
- Activation functions

The network also contains all three optional/enhanced components requested for
this version:

- Batch Normalization
- Dropout
- Residual Connection

The main training script supports comparison experiments required by the PDF:

- Different numbers of filters: `baseline` vs `wider_filters`
- Different activations: `relu` vs `gelu`
- Different loss functions / regularization settings:
  standard cross entropy, label-smoothed cross entropy, focal loss, MSE loss,
  dropout, and weight decay

The script records and saves:

- Best accuracy
- Test error
- Train loss / validation loss
- Train accuracy / validation accuracy
- Parameter count
- Best model weights
- Last model weights
- Training speed per epoch

The script also produces visualizations:

- Training curves
- Confusion matrix
- First-layer convolution filter visualization

Main related files:

```text
models/residual_cnn.py
train_residual_cifar10.py
```

### 2. Self-Implemented Optimizer

The PDF allows choosing the most difficult optimizer option:

> Implement an optimizer for your full model by yourself.

This is implemented as `ManualAdamW`.

It does not subclass or call `torch.optim`. It manually performs:

- First moment update
- Second moment update
- Bias correction
- Decoupled weight decay
- Parameter update for the full model

The main CIFAR-10 model is trained with `ManualAdamW` by default.

Main related file:

```text
optimizers/manual_adamw.py
```

### 3. Batch Normalization (30%)

Implemented `VGG_A_BatchNorm` based on the provided VGG-A model.

The BN experiment compares:

- VGG-A without BatchNorm
- VGG-A with BatchNorm

The experiment supports the learning rates required in the PDF:

```text
[1e-3, 2e-3, 1e-4, 5e-4]
```

The loss landscape workflow records per-step losses, computes:

- `max_curve`
- `min_curve`
- `mean_curve`

It then uses `matplotlib.pyplot.fill_between` to visualize the loss envelope
for VGG-A and VGG-A+BN in the same figure.

Extra gradient analysis is also included to support the explanation of how BN
helps optimization:

- Gradient norm comparison
- Gradient change comparison
- Maximum gradient change summary

Main related files:

```text
models/vgg.py
VGG_Loss_Landscape.py
```

## What Is Not Done Yet

The implementation and smoke tests are complete, but the following final
submission tasks still need to be done after formal training:

- Run full CIFAR-10 experiments
- Record the final best test error
- Choose final figures for the report
- Upload code to GitHub
- Upload trained model weights
- Provide dataset and model-weight links in the report
- Write the final PDF report with name and student ID
- Submit the PDF report through elearning before `2026-06-14 23:59`

## How to Run

Enter the project code directory first:

```powershell
cd "S:\study\大三下\神经网络\作业\hw3\PJ2_2026\codes\VGG_BatchNorm"
```

### 1. Fast Sanity Check

Run this first:

```powershell
python sanity_checks.py
```

This verifies:

- ManualAdamW matches `torch.optim.AdamW` on a tiny model
- `ResidualCifarNet` can run forward / backward / update
- `VGG_A_BatchNorm` can run a forward pass

### 2. Quick Smoke Tests Without CIFAR-10

Main model smoke test:

```powershell
python train_residual_cifar10.py --synthetic --epochs 1 --batch-size 16 --n-train 32 --n-val 16
```

BatchNorm experiment smoke test:

```powershell
python VGG_Loss_Landscape.py --synthetic --epochs 1 --learning-rates 0.001 --batch-size 2 --n-train 4 --n-val 4
```

### 3. Formal Main CIFAR-10 Training

Train the final residual CNN with BatchNorm, Dropout, residual connections, and
the manual AdamW optimizer:

```powershell
python train_residual_cifar10.py --download --epochs 30 --batch-size 128
```

Outputs are saved to:

```text
reports/residual_cifar/
```

Important outputs include:

- `best_residual_cifar.pt`
- `last_residual_cifar.pt`
- `history.json`
- `config.json`
- `training_curves.png`
- `confusion_matrix.png`
- `first_layer_filters.png`

### 4. CIFAR-10 Ablation Suite

Run the comparison experiments for filters, activations, loss functions, and
regularization:

```powershell
python train_residual_cifar10.py --download --suite --epochs 30 --batch-size 128
```

Outputs are saved to:

```text
reports/residual_cifar/
```

Each preset gets its own subdirectory.

The suite presets are:

- `baseline`: standard cross entropy, ReLU, base channels 32
- `wider_filters`: standard cross entropy, base channels 48
- `gelu_activation`: standard cross entropy, GELU activation
- `strong_regularization`: label-smoothed cross entropy, stronger dropout and weight decay
- `focal_loss`: focal loss with `gamma=2`
- `mse_loss`: MSE between softmax probabilities and one-hot labels

If the first four presets have already been trained, run only the two new loss
experiments:

```powershell
python train_residual_cifar10.py --download --preset focal_loss --epochs 30 --batch-size 128 --output-dir reports/residual_cifar/focal_loss
python train_residual_cifar10.py --download --preset mse_loss --epochs 30 --batch-size 128 --output-dir reports/residual_cifar/mse_loss
```

### 5. BatchNorm / Loss Landscape Experiment

Run VGG-A and VGG-A+BN across the required learning-rate list:

```powershell
python VGG_Loss_Landscape.py --download --epochs 10 --n-train 4096 --n-val 1024
```

Outputs are saved to:

```text
reports/vgg_bn/
```

Important outputs include:

- `loss_landscape_bn_vs_no_bn.png`
- `gradient_norms_bn_vs_no_bn.png`
- `gradient_changes_bn_vs_no_bn.png`
- `vgg_a_min_curve.txt`
- `vgg_a_max_curve.txt`
- `vgg_a_bn_min_curve.txt`
- `vgg_a_bn_max_curve.txt`
- Per-run `.json` history files
- Per-run `.pt` model weights

## Suggested Report Structure

Use the generated results to write the final PDF report in this order:

1. Basic information: name, student ID, GitHub link, dataset link, model-weight link
2. CIFAR-10 model architecture
3. Manual AdamW optimizer implementation
4. CIFAR-10 experimental setup
5. CIFAR-10 results and visualization
6. VGG-A vs VGG-A+BN setup
7. Loss landscape and gradient comparison
8. Main findings and conclusion

## Verification Already Passed

The following checks have already been run successfully:

```powershell
python sanity_checks.py
python train_residual_cifar10.py --synthetic --epochs 1 --batch-size 16 --n-train 32 --n-val 16
python VGG_Loss_Landscape.py --synthetic --epochs 1 --learning-rates 0.001 --batch-size 2 --n-train 4 --n-val 4
python -m compileall .
```
