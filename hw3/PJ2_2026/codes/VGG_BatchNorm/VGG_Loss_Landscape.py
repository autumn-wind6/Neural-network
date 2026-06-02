"""
VGG-A BatchNorm comparison and loss-landscape visualization.

This script trains VGG-A and VGG-A+BN across several learning rates, records
per-step losses and gradient norms, and plots the min/max loss envelope required
by the assignment.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data.loaders import get_cifar_loader
from models.vgg import VGG_A, VGG_A_BatchNorm, get_number_of_parameters
from optimizers import ManualAdamW


def parse_args():
    parser = argparse.ArgumentParser(description="Compare VGG-A with and without BN")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "reports" / "vgg_bn")
    parser.add_argument("--download", action="store_true", help="Allow torchvision to download CIFAR-10")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data for quick checks")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--n-train", type=int, default=4096)
    parser.add_argument("--n-val", type=int, default=1024)
    parser.add_argument("--learning-rates", nargs="+", type=float,
                        default=[1e-3, 2e-3, 1e-4, 5e-4])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


class SyntheticCIFAR10(Dataset):
    def __init__(self, n_items=256, seed=0):
        generator = torch.Generator().manual_seed(seed)
        self.x = torch.rand(n_items, 3, 32, 32, generator=generator) * 2 - 1
        self.y = torch.randint(0, 10, (n_items,), generator=generator)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.y)


def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1)


def gradient_norm(model):
    total = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        total += param.grad.detach().pow(2).sum().item()
    return float(np.sqrt(total))


def train(model, optimizer, criterion, train_loader, val_loader, device,
          epochs_n=10):
    model.to(device)
    history = {
        "step_losses": [],
        "step_grad_norms": [],
        "epoch_train_loss": [],
        "epoch_val_accuracy": [],
        "epoch_seconds": [],
    }

    for epoch in range(1, epochs_n + 1):
        start = time.perf_counter()
        model.train()
        running_loss = 0.0
        running_items = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            history["step_losses"].append(float(loss.item()))
            history["step_grad_norms"].append(gradient_norm(model))
            optimizer.step()

            running_loss += loss.item() * y.size(0)
            running_items += y.size(0)

        val_acc = get_accuracy(model, val_loader, device)
        history["epoch_train_loss"].append(running_loss / max(running_items, 1))
        history["epoch_val_accuracy"].append(val_acc)
        history["epoch_seconds"].append(time.perf_counter() - start)
        print(
            f"epoch {epoch:03d}/{epochs_n} "
            f"train_loss={history['epoch_train_loss'][-1]:.4f} "
            f"val_acc={val_acc:.4f}"
        )

    return history


def save_json(path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def run_single(model_name, model_factory, lr, args, train_loader, val_loader, device):
    set_random_seeds(args.seed)
    model = model_factory()
    optimizer = ManualAdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    print(f"\n=== {model_name}, lr={lr:g}, params={get_number_of_parameters(model)} ===")
    history = train(
        model,
        optimizer,
        criterion,
        train_loader,
        val_loader,
        device,
        epochs_n=args.epochs,
    )
    history["lr"] = lr
    history["model"] = model_name
    return model, history


def build_loss_envelope(histories):
    min_len = min(len(history["step_losses"]) for history in histories)
    losses = np.array([history["step_losses"][:min_len] for history in histories])
    return {
        "min_curve": losses.min(axis=0),
        "max_curve": losses.max(axis=0),
        "mean_curve": losses.mean(axis=0),
    }


def plot_loss_landscape(no_bn_envelope, bn_envelope, output_dir):
    fig, ax = plt.subplots(figsize=(9, 5))
    steps = np.arange(len(no_bn_envelope["min_curve"]))

    ax.fill_between(
        steps,
        no_bn_envelope["min_curve"],
        no_bn_envelope["max_curve"],
        alpha=0.25,
        label="VGG-A loss range",
    )
    ax.plot(steps, no_bn_envelope["mean_curve"], label="VGG-A mean")
    ax.fill_between(
        steps,
        bn_envelope["min_curve"],
        bn_envelope["max_curve"],
        alpha=0.25,
        label="VGG-A+BN loss range",
    )
    ax.plot(steps, bn_envelope["mean_curve"], label="VGG-A+BN mean")
    ax.set_xlabel("training step")
    ax.set_ylabel("cross entropy loss")
    ax.set_title("Loss landscape envelope across learning rates")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "loss_landscape_bn_vs_no_bn.png", dpi=180)
    plt.close(fig)


def plot_gradient_norms(no_bn_histories, bn_histories, output_dir):
    fig, ax = plt.subplots(figsize=(9, 5))
    for history in no_bn_histories:
        ax.plot(history["step_grad_norms"], alpha=0.45,
                label=f"VGG-A lr={history['lr']:g}")
    for history in bn_histories:
        ax.plot(history["step_grad_norms"], alpha=0.45,
                linestyle="--", label=f"VGG-A+BN lr={history['lr']:g}")
    ax.set_xlabel("training step")
    ax.set_ylabel("gradient norm")
    ax.set_title("Gradient norm comparison")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "gradient_norms_bn_vs_no_bn.png", dpi=180)
    plt.close(fig)


def gradient_change_curve(histories):
    changes = []
    for history in histories:
        grad_norms = np.array(history["step_grad_norms"], dtype=np.float64)
        if len(grad_norms) > 1:
            changes.append(np.abs(np.diff(grad_norms)))
    if not changes:
        return np.array([])
    min_len = min(len(change) for change in changes)
    stacked = np.array([change[:min_len] for change in changes])
    return stacked.mean(axis=0)


def plot_gradient_changes(no_bn_histories, bn_histories, output_dir):
    no_bn_change = gradient_change_curve(no_bn_histories)
    bn_change = gradient_change_curve(bn_histories)
    if len(no_bn_change) == 0 or len(bn_change) == 0:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(no_bn_change, label="VGG-A mean |delta grad norm|")
    ax.plot(bn_change, label="VGG-A+BN mean |delta grad norm|")
    ax.set_xlabel("training step")
    ax.set_ylabel("mean absolute gradient-norm change")
    ax.set_title("Gradient change comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "gradient_changes_bn_vs_no_bn.png", dpi=180)
    plt.close(fig)


def max_gradient_change(histories):
    max_change = 0.0
    for history in histories:
        grad_norms = np.array(history["step_grad_norms"], dtype=np.float64)
        if len(grad_norms) > 1:
            max_change = max(max_change, float(np.abs(np.diff(grad_norms)).max()))
    return max_change


def save_envelope_txt(envelope, prefix, output_dir):
    np.savetxt(output_dir / f"{prefix}_min_curve.txt", envelope["min_curve"])
    np.savetxt(output_dir / f"{prefix}_max_curve.txt", envelope["max_curve"])
    np.savetxt(output_dir / f"{prefix}_mean_curve.txt", envelope["mean_curve"])


def main():
    args = parse_args()
    set_random_seeds(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if args.synthetic:
        n_train = args.n_train if args.n_train > 0 else 256
        n_val = args.n_val if args.n_val > 0 else 128
        train_loader = DataLoader(
            SyntheticCIFAR10(n_train, seed=args.seed),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            SyntheticCIFAR10(n_val, seed=args.seed + 1),
            batch_size=args.batch_size,
            shuffle=False,
        )
    else:
        train_loader = get_cifar_loader(
            root=args.data_dir,
            batch_size=args.batch_size,
            train=True,
            shuffle=True,
            num_workers=args.num_workers,
            n_items=args.n_train,
            download=args.download,
            augment=True,
            drop_last=True,
        )
        val_loader = get_cifar_loader(
            root=args.data_dir,
            batch_size=args.batch_size,
            train=False,
            shuffle=False,
            num_workers=args.num_workers,
            n_items=args.n_val,
            download=args.download,
            augment=False,
        )

    no_bn_histories = []
    bn_histories = []
    summary = {
        "learning_rates": args.learning_rates,
        "epochs": args.epochs,
        "n_train": args.n_train,
        "n_val": args.n_val,
        "optimizer": "ManualAdamW",
        "device": str(device),
    }

    for lr in args.learning_rates:
        model, history = run_single(
            "vgg_a", VGG_A, lr, args, train_loader, val_loader, device
        )
        no_bn_histories.append(history)
        torch.save(model.state_dict(), args.output_dir / f"vgg_a_lr_{lr:g}.pt")
        save_json(args.output_dir / f"vgg_a_lr_{lr:g}.json", history)

        model_bn, history_bn = run_single(
            "vgg_a_bn", VGG_A_BatchNorm, lr, args, train_loader, val_loader, device
        )
        bn_histories.append(history_bn)
        torch.save(model_bn.state_dict(), args.output_dir / f"vgg_a_bn_lr_{lr:g}.pt")
        save_json(args.output_dir / f"vgg_a_bn_lr_{lr:g}.json", history_bn)

    no_bn_envelope = build_loss_envelope(no_bn_histories)
    bn_envelope = build_loss_envelope(bn_histories)
    plot_loss_landscape(no_bn_envelope, bn_envelope, args.output_dir)
    plot_gradient_norms(no_bn_histories, bn_histories, args.output_dir)
    plot_gradient_changes(no_bn_histories, bn_histories, args.output_dir)
    save_envelope_txt(no_bn_envelope, "vgg_a", args.output_dir)
    save_envelope_txt(bn_envelope, "vgg_a_bn", args.output_dir)

    summary["vgg_a_final_val_acc"] = [
        history["epoch_val_accuracy"][-1] for history in no_bn_histories
    ]
    summary["vgg_a_bn_final_val_acc"] = [
        history["epoch_val_accuracy"][-1] for history in bn_histories
    ]
    summary["vgg_a_max_gradient_change"] = max_gradient_change(no_bn_histories)
    summary["vgg_a_bn_max_gradient_change"] = max_gradient_change(bn_histories)
    save_json(args.output_dir / "summary.json", summary)
    print(f"\nartifacts saved to {args.output_dir}")


if __name__ == "__main__":
    main()
