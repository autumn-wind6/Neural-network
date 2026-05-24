import os
import gzip
import argparse
from struct import unpack

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import mynn as nn


def load_mnist(images_path, labels_path):
    with gzip.open(images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)
    with gzip.open(labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return images / 255.0, labels


def load_model(model_name, model_path):
    if model_name == 'MLP':
        model = nn.models.Model_MLP()
    elif model_name == 'CNN':
        model = nn.models.Model_CNN()
    else:
        raise ValueError(f'Unknown model_name: {model_name}')
    model.load_model(model_path)
    if hasattr(model, 'eval'):
        model.eval()
    return model


def predict(model, images, batch_size=256):
    logits_list = []
    for start in range(0, images.shape[0], batch_size):
        logits_list.append(model(images[start:start + batch_size]))
    logits = np.concatenate(logits_list, axis=0)
    return logits, np.argmax(logits, axis=1)


def confusion_matrix(labels, preds, num_classes=10):
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for y_true, y_pred in zip(labels, preds):
        matrix[y_true, y_pred] += 1
    return matrix


def plot_confusion_matrix(matrix, save_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap='Blues')
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = 'white' if matrix[i, j] > matrix.max() * 0.55 else 'black'
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center', fontsize=8, color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_misclassified(images, labels, preds, save_path, max_examples=25):
    wrong_indices = np.where(labels != preds)[0][:max_examples]
    if wrong_indices.size == 0:
        return
    cols = 5
    rows = int(np.ceil(wrong_indices.size / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 2.0))
    axes = np.asarray(axes).reshape(-1)
    for ax in axes:
        ax.axis('off')
    for ax, idx in zip(axes, wrong_indices):
        ax.imshow(images[idx].reshape(28, 28), cmap='gray')
        ax.set_title(f'T:{labels[idx]} P:{preds[idx]}', fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_mlp_weights(model, save_path, max_units=25):
    first_linear = next(layer for layer in model.layers if layer.optimizable)
    weights = first_linear.params['W']
    cols = 5
    count = min(max_units, weights.shape[1])
    rows = int(np.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 1.8))
    axes = np.asarray(axes).reshape(-1)
    for ax in axes:
        ax.axis('off')
    for i in range(count):
        axes[i].imshow(weights[:, i].reshape(28, 28), cmap='coolwarm')
        axes[i].set_title(f'unit {i}', fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_cnn_kernels(model, save_path):
    conv_layer = next(layer for layer in model.layers if layer.__class__.__name__ == 'conv2D')
    kernels = conv_layer.params['W']
    count = kernels.shape[0]
    cols = min(8, count)
    rows = int(np.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.6))
    axes = np.asarray(axes).reshape(-1)
    for ax in axes:
        ax.axis('off')
    for i in range(count):
        kernel = kernels[i, 0]
        axes[i].imshow(kernel, cmap='coolwarm')
        axes[i].set_title(f'k{i}', fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['MLP', 'CNN'], default='MLP')
    parser.add_argument('--model-path', default=None)
    parser.add_argument('--output-dir', default=r'.\figs')
    parser.add_argument('--num-misclassified', type=int, default=25)
    args = parser.parse_args()

    model_name = args.model
    model_path = args.model_path or rf'.\best_models\{model_name.lower()}_baseline\best_model.pickle'
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)
    model = load_model(model_name, model_path)
    test_imgs, test_labs = load_mnist(
        r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz',
        r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz',
    )
    logits, preds = predict(model, test_imgs)
    accuracy = nn.metric.accuracy(logits, test_labs)
    matrix = confusion_matrix(test_labs, preds)

    plot_confusion_matrix(matrix, os.path.join(output_dir, f'{model_name}_confusion_matrix.png'))
    plot_misclassified(
        test_imgs,
        test_labs,
        preds,
        os.path.join(output_dir, f'{model_name}_misclassified.png'),
        args.num_misclassified,
    )
    if model_name == 'MLP':
        plot_mlp_weights(model, os.path.join(output_dir, f'{model_name}_weights.png'))
    else:
        plot_cnn_kernels(model, os.path.join(output_dir, f'{model_name}_kernels.png'))

    print(f'{model_name} test accuracy: {accuracy:.6f}')
    print(f'outputs saved to: {output_dir}')


if __name__ == '__main__':
    main()
