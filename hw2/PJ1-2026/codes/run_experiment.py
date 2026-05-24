import argparse
import gzip
import json
import os
from struct import unpack

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import mynn as nn


def load_images(path):
    with gzip.open(path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)


def load_labels(path):
    with gzip.open(path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)


def load_data(seed, valid_size, train_limit=None, valid_limit=None, test_limit=None):
    train_imgs = load_images(r'.\dataset\MNIST\train-images-idx3-ubyte.gz')
    train_labs = load_labels(r'.\dataset\MNIST\train-labels-idx1-ubyte.gz')
    test_imgs = load_images(r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz')
    test_labs = load_labels(r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz')

    rng = np.random.RandomState(seed)
    idx = rng.permutation(np.arange(train_imgs.shape[0]))
    train_imgs = train_imgs[idx]
    train_labs = train_labs[idx]

    valid_imgs = train_imgs[:valid_size]
    valid_labs = train_labs[:valid_size]
    train_imgs = train_imgs[valid_size:]
    train_labs = train_labs[valid_size:]

    train_imgs = train_imgs / 255.0
    valid_imgs = valid_imgs / 255.0
    test_imgs = test_imgs / 255.0

    if train_limit is not None:
        train_imgs = train_imgs[:train_limit]
        train_labs = train_labs[:train_limit]
    if valid_limit is not None:
        valid_imgs = valid_imgs[:valid_limit]
        valid_labs = valid_labs[:valid_limit]
    if test_limit is not None:
        test_imgs = test_imgs[:test_limit]
        test_labs = test_labs[:test_limit]

    return (train_imgs, train_labs), (valid_imgs, valid_labs), (test_imgs, test_labs)


def build_model(model_name, experiment, hidden_dim, conv_channels, dropout_rate):
    use_dropout = experiment == 'dropout'

    if model_name == 'MLP':
        dropout_list = [dropout_rate] if use_dropout else None
        return nn.models.Model_MLP([784, hidden_dim, 10], 'ReLU', None, dropout_list)
    if model_name == 'CNN':
        cnn_dropout = dropout_rate if use_dropout else 0.0
        return nn.models.Model_CNN(conv_channels=conv_channels, dropout_rate=cnn_dropout)
    raise ValueError(f'Unknown model: {model_name}')


def load_best_model(model_name, model_path):
    if model_name == 'MLP':
        model = nn.models.Model_MLP()
    elif model_name == 'CNN':
        model = nn.models.Model_CNN()
    else:
        raise ValueError(f'Unknown model: {model_name}')
    model.load_model(model_path)
    if hasattr(model, 'eval'):
        model.eval()
    return model


def save_learning_curve(runner, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.set_tight_layout(True)
    train_steps = np.arange(len(runner.train_loss))
    dev_steps = np.linspace(0, max(len(runner.train_loss) - 1, 0), num=len(runner.dev_loss))
    axes[0].plot(train_steps, runner.train_loss, label='Train loss')
    axes[0].plot(dev_steps, runner.dev_loss, linestyle='--', label='Dev loss')
    axes[0].set_xlabel('iteration')
    axes[0].set_ylabel('loss')
    axes[0].legend(loc='upper right')

    train_score_steps = np.arange(len(runner.train_scores))
    dev_score_steps = np.linspace(0, max(len(runner.train_scores) - 1, 0), num=len(runner.dev_scores))
    axes[1].plot(train_score_steps, runner.train_scores, label='Train accuracy')
    axes[1].plot(dev_score_steps, runner.dev_scores, linestyle='--', label='Dev accuracy')
    axes[1].set_xlabel('iteration')
    axes[1].set_ylabel('accuracy')
    axes[1].legend(loc='lower right')
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['MLP', 'CNN'], required=True)
    parser.add_argument('--experiment', choices=['baseline', 'dropout'], default='baseline')
    parser.add_argument('--run-name', default=None)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.06)
    parser.add_argument('--hidden-dim', type=int, default=600)
    parser.add_argument('--conv-channels', type=int, default=8)
    parser.add_argument('--dropout-rate', type=float, default=0.5)
    parser.add_argument('--valid-size', type=int, default=10000)
    parser.add_argument('--log-iters', type=int, default=100)
    parser.add_argument('--eval-iters', type=int, default=200)
    parser.add_argument('--seed', type=int, default=309)
    parser.add_argument('--train-limit', type=int, default=None)
    parser.add_argument('--valid-limit', type=int, default=None)
    parser.add_argument('--test-limit', type=int, default=None)
    args = parser.parse_args()

    np.random.seed(args.seed)
    run_name = args.run_name or f'{args.model.lower()}_{args.experiment}'
    save_dir = os.path.join('best_models', run_name)
    os.makedirs(save_dir, exist_ok=True)

    train_set, valid_set, test_set = load_data(
        args.seed,
        args.valid_size,
        train_limit=args.train_limit,
        valid_limit=args.valid_limit,
        test_limit=args.test_limit,
    )
    model = build_model(
        args.model,
        args.experiment,
        args.hidden_dim,
        args.conv_channels,
        args.dropout_rate,
    )
    optimizer = nn.optimizer.SGD(init_lr=args.lr, model=model)
    loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=10)
    runner = nn.runner.RunnerM(model, optimizer, nn.metric.accuracy, loss_fn, batch_size=args.batch_size)

    runner.train(
        train_set,
        valid_set,
        num_epochs=args.epochs,
        log_iters=args.log_iters,
        eval_iters=args.eval_iters,
        save_dir=save_dir,
    )

    model_path = os.path.join(save_dir, 'best_model.pickle')
    best_model = load_best_model(args.model, model_path)
    test_loss_fn = nn.op.MultiCrossEntropyLoss(model=best_model, max_classes=10)
    test_runner = nn.runner.RunnerM(best_model, optimizer, nn.metric.accuracy, test_loss_fn, batch_size=args.batch_size)
    test_score, test_loss = test_runner.evaluate(test_set)

    learning_curve_path = os.path.join(save_dir, 'learning_curve.png')
    save_learning_curve(runner, learning_curve_path)
    figs_learning_curve_dir = os.path.join('figs', 'learning_curves')
    os.makedirs(figs_learning_curve_dir, exist_ok=True)
    figs_learning_curve_path = os.path.join(figs_learning_curve_dir, f'{run_name}_learning_curve.png')
    save_learning_curve(runner, figs_learning_curve_path)
    metrics = {
        'run_name': run_name,
        'model': args.model,
        'experiment': args.experiment,
        'best_dev_accuracy': float(runner.best_score),
        'test_accuracy': float(test_score),
        'test_loss': float(test_loss),
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'hidden_dim': args.hidden_dim if args.model == 'MLP' else None,
        'conv_channels': args.conv_channels if args.model == 'CNN' else None,
        'dropout_rate': args.dropout_rate if args.experiment == 'dropout' else None,
        'model_path': model_path,
        'learning_curve': learning_curve_path,
        'figs_learning_curve': figs_learning_curve_path,
    }
    with open(os.path.join(save_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
