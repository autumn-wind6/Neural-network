"""
Fast checks for Project 2 code.

These checks do not require CIFAR-10 files. They verify:
1. ManualAdamW closely matches torch.optim.AdamW on a tiny model.
2. ResidualCifarNet can run a forward/backward/update step.
3. VGG_A_BatchNorm can run a forward pass.
"""
import copy
import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from models.residual_cnn import ResidualCifarNet
from models.vgg import VGG_A_BatchNorm
from optimizers import ManualAdamW


def check_manual_adamw_matches_torch():
    torch.manual_seed(7)
    model_manual = nn.Sequential(
        nn.Linear(5, 8),
        nn.ReLU(),
        nn.Linear(8, 3),
    )
    model_torch = copy.deepcopy(model_manual)
    x = torch.randn(11, 5)
    y = torch.randn(11, 3)
    criterion = nn.MSELoss()

    manual_optimizer = ManualAdamW(
        model_manual.parameters(),
        lr=1e-3,
        weight_decay=1e-2,
    )
    try:
        torch_optimizer = torch.optim.AdamW(
            model_torch.parameters(),
            lr=1e-3,
            weight_decay=1e-2,
            foreach=False,
        )
    except TypeError:
        torch_optimizer = torch.optim.AdamW(
            model_torch.parameters(),
            lr=1e-3,
            weight_decay=1e-2,
        )

    for _ in range(3):
        manual_optimizer.zero_grad()
        torch_optimizer.zero_grad()

        manual_loss = criterion(model_manual(x), y)
        torch_loss = criterion(model_torch(x), y)
        manual_loss.backward()
        torch_loss.backward()
        manual_optimizer.step()
        torch_optimizer.step()

    max_diff = 0.0
    for manual_param, torch_param in zip(model_manual.parameters(), model_torch.parameters()):
        max_diff = max(max_diff, (manual_param - torch_param).abs().max().item())

    print(f"ManualAdamW max parameter diff vs torch AdamW: {max_diff:.3e}")
    assert max_diff < 1e-6


def check_residual_model_step():
    torch.manual_seed(11)
    model = ResidualCifarNet(base_channels=8, dropout=0.1)
    optimizer = ManualAdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    x = torch.randn(4, 3, 32, 32)
    y = torch.tensor([0, 1, 2, 3])

    optimizer.zero_grad()
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()

    print(f"ResidualCifarNet logits shape: {tuple(logits.shape)}, loss={loss.item():.4f}")
    assert logits.shape == (4, 10)
    assert torch.isfinite(loss)


def check_vgg_bn_forward():
    torch.manual_seed(13)
    model = VGG_A_BatchNorm()
    x = torch.randn(2, 3, 32, 32)
    logits = model(x)

    print(f"VGG_A_BatchNorm logits shape: {tuple(logits.shape)}")
    assert logits.shape == (2, 10)
    assert torch.isfinite(logits).all()


def main():
    check_manual_adamw_matches_torch()
    check_residual_model_step()
    check_vgg_bn_forward()
    print("All sanity checks passed.")


if __name__ == "__main__":
    main()
