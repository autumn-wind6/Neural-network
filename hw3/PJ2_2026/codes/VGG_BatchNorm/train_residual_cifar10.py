"""
Main CIFAR-10 experiment for Project 2.

The default model contains BatchNorm, Dropout and residual connections, and the
training loop uses ManualAdamW instead of torch.optim.
"""
import argparse
import copy
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
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from models.residual_cnn import ResidualCifarNet, count_parameters
from optimizers import ManualAdamW


CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

PRESETS = {
    "baseline": {
        "base_channels": 32,
        "activation": "relu",
        "dropout": 0.20,
        "loss_function": "cross_entropy",
        "label_smoothing": 0.00,
        "weight_decay": 5e-4,
    },
    "wider_filters": {
        "base_channels": 48,
        "activation": "relu",
        "dropout": 0.20,
        "loss_function": "cross_entropy",
        "label_smoothing": 0.00,
        "weight_decay": 5e-4,
    },
    "gelu_activation": {
        "base_channels": 32,
        "activation": "gelu",
        "dropout": 0.20,
        "loss_function": "cross_entropy",
        "label_smoothing": 0.00,
        "weight_decay": 5e-4,
    },
    "strong_regularization": {
        "base_channels": 32,
        "activation": "relu",
        "dropout": 0.35,
        "loss_function": "cross_entropy",
        "label_smoothing": 0.10,
        "weight_decay": 1e-3,
    },
    "focal_loss": {
        "base_channels": 32,
        "activation": "relu",
        "dropout": 0.20,
        "loss_function": "focal",
        "label_smoothing": 0.00,
        "focal_gamma": 2.0,
        "weight_decay": 5e-4,
    },
    "mse_loss": {
        "base_channels": 32,
        "activation": "relu",
        "dropout": 0.20,
        "loss_function": "mse",
        "label_smoothing": 0.00,
        "weight_decay": 5e-4,
    },
}


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(
            logits,
            targets,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        probs = nn.functional.softmax(logits, dim=1)
        target_probs = probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - target_probs).pow(self.gamma)
        return (focal_weight * ce_loss).mean()


class MSEClassificationLoss(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits, targets):
        probs = nn.functional.softmax(logits, dim=1)
        one_hot = nn.functional.one_hot(targets, num_classes=self.num_classes).float()
        return nn.functional.mse_loss(probs, one_hot)


class SyntheticCIFAR10(Dataset):
    def __init__(self, n_items=256, seed=0):
        generator = torch.Generator().manual_seed(seed)
        self.x = torch.rand(n_items, 3, 32, 32, generator=generator) * 2 - 1
        self.y = torch.randint(0, 10, (n_items,), generator=generator)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.y)


def parse_args():
    parser = argparse.ArgumentParser(description="Train ResidualCifarNet on CIFAR-10")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "reports" / "residual_cifar")
    parser.add_argument("--download", action="store_true", help="Allow torchvision to download CIFAR-10")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data for quick checks")
    parser.add_argument("--suite", action="store_true", help="Run all comparison presets")
    parser.add_argument("--preset", choices=["custom"] + list(PRESETS), default="custom")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--n-train", type=int, default=-1, help="Limit train items; -1 uses all")
    parser.add_argument("--n-val", type=int, default=-1, help="Limit test items; -1 uses all")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default=None)

    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--activation", choices=["relu", "gelu", "silu", "leaky_relu"], default="relu")
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--loss-function", choices=["cross_entropy", "focal", "mse"], default="cross_entropy")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    return parser.parse_args()


def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def cifar_transform(train=True):
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    steps = []
    if train:
        steps.extend([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
    steps.extend([transforms.ToTensor(), normalize])
    return transforms.Compose(steps)


def limit_dataset(dataset, n_items):
    if n_items is None or n_items < 0:
        return dataset
    return Subset(dataset, range(min(n_items, len(dataset))))


def make_loaders(args):
    if args.synthetic:
        n_train = args.n_train if args.n_train > 0 else 256
        n_val = args.n_val if args.n_val > 0 else 128
        train_set = SyntheticCIFAR10(n_train, seed=args.seed)
        val_set = SyntheticCIFAR10(n_val, seed=args.seed + 1)
    else:
        train_set = datasets.CIFAR10(
            root=str(args.data_dir),
            train=True,
            download=args.download,
            transform=cifar_transform(train=True),
        )
        val_set = datasets.CIFAR10(
            root=str(args.data_dir),
            train=False,
            download=args.download,
            transform=cifar_transform(train=False),
        )
        train_set = limit_dataset(train_set, args.n_train)
        val_set = limit_dataset(val_set, args.n_val)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def accuracy_from_logits(logits, targets):
    preds = logits.argmax(dim=1)
    return (preds == targets).sum().item(), preds


def make_criterion(args):
    if args.loss_function == "cross_entropy":
        return nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    if args.loss_function == "focal":
        return FocalLoss(
            gamma=args.focal_gamma,
            label_smoothing=args.label_smoothing,
        )
    if args.loss_function == "mse":
        return MSEClassificationLoss(num_classes=len(CLASS_NAMES))
    raise ValueError(f"Unsupported loss function: {args.loss_function}")


def evaluate(model, loader, criterion, device, collect_outputs=False):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            correct, preds = accuracy_from_logits(logits, y)
            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            total_correct += correct
            total_items += batch_size
            if collect_outputs:
                all_preds.append(preds.cpu())
                all_targets.append(y.cpu())

    result = {
        "loss": total_loss / max(total_items, 1),
        "accuracy": total_correct / max(total_items, 1),
    }
    if collect_outputs:
        result["preds"] = torch.cat(all_preds).numpy()
        result["targets"] = torch.cat(all_targets).numpy()
    return result


def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip=0.0):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_items = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        correct, _ = accuracy_from_logits(logits.detach(), y)
        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_correct += correct
        total_items += batch_size

    return {
        "loss": total_loss / max(total_items, 1),
        "accuracy": total_correct / max(total_items, 1),
    }


def plot_curves(history, output_dir):
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_accuracy"], label="train")
    axes[1].plot(epochs, history["val_accuracy"], label="val")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=180)
    plt.close(fig)


def plot_confusion_matrix(targets, preds, output_dir):
    matrix = np.zeros((10, 10), dtype=np.int64)
    for target, pred in zip(targets, preds):
        matrix[target, pred] += 1

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(10), CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(10), CLASS_NAMES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("CIFAR-10 confusion matrix")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)


def plot_first_layer_filters(model, output_dir, max_filters=16):
    weights = model.first_conv_weight()
    n_filters = min(max_filters, weights.size(0))
    cols = 4
    rows = int(np.ceil(n_filters / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = np.asarray(axes).reshape(-1)

    for index in range(n_filters):
        filt = weights[index]
        filt = filt - filt.min()
        filt = filt / (filt.max() + 1e-12)
        axes[index].imshow(filt.permute(1, 2, 0))
        axes[index].axis("off")
    for index in range(n_filters, len(axes)):
        axes[index].axis("off")

    fig.suptitle("First convolution filters")
    fig.tight_layout()
    fig.savefig(output_dir / "first_layer_filters.png", dpi=180)
    plt.close(fig)


def save_json(path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def apply_preset(args, preset_name):
    run_args = copy.deepcopy(args)
    if preset_name != "custom":
        for key, value in PRESETS[preset_name].items():
            setattr(run_args, key, value)
    run_args.preset = preset_name
    return run_args


def train_experiment(args):
    set_random_seeds(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = make_loaders(args)
    model = ResidualCifarNet(
        base_channels=args.base_channels,
        activation=args.activation,
        dropout=args.dropout,
    ).to(device)
    criterion = make_criterion(args)
    optimizer = ManualAdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    config = vars(copy.deepcopy(args))
    config["data_dir"] = str(config["data_dir"])
    config["output_dir"] = str(config["output_dir"])
    config["parameter_count"] = count_parameters(model)
    save_json(output_dir / "config.json", config)

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "epoch_seconds": [],
    }
    best_accuracy = -1.0
    best_epoch = 0
    best_path = output_dir / "best_residual_cifar.pt"

    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, grad_clip=args.grad_clip
        )
        val_metrics = evaluate(model, val_loader, criterion, device)
        elapsed = time.perf_counter() - start

        history["train_loss"].append(train_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["epoch_seconds"].append(elapsed)

        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": config,
                "epoch": epoch,
                "val_accuracy": best_accuracy,
                "test_error": 1 - best_accuracy,
            }, best_path)

        print(
            f"epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"time={elapsed:.1f}s"
        )

    final_eval = evaluate(model, val_loader, criterion, device, collect_outputs=True)
    history["best_epoch"] = best_epoch
    history["best_val_accuracy"] = best_accuracy
    history["best_test_error"] = 1 - best_accuracy
    history["final_val_accuracy"] = final_eval["accuracy"]
    history["final_test_error"] = 1 - final_eval["accuracy"]
    save_json(output_dir / "history.json", history)

    plot_curves(history, output_dir)
    plot_confusion_matrix(final_eval["targets"], final_eval["preds"], output_dir)
    plot_first_layer_filters(model, output_dir)

    torch.save({
        "model_state": model.state_dict(),
        "config": config,
        "history": history,
    }, output_dir / "last_residual_cifar.pt")
    print(f"best accuracy={best_accuracy:.4f}, test error={1 - best_accuracy:.4f}")
    print(f"artifacts saved to {output_dir}")
    return history


def main():
    args = parse_args()
    if args.suite:
        suite_summary = {}
        for preset_name in PRESETS:
            run_args = apply_preset(args, preset_name)
            run_args.output_dir = args.output_dir / preset_name
            print(f"\n=== running preset: {preset_name} ===")
            history = train_experiment(run_args)
            suite_summary[preset_name] = {
                "best_val_accuracy": history["best_val_accuracy"],
                "best_test_error": history["best_test_error"],
                "best_epoch": history["best_epoch"],
            }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        save_json(args.output_dir / "suite_summary.json", suite_summary)
    else:
        train_experiment(apply_preset(args, args.preset))


if __name__ == "__main__":
    main()
