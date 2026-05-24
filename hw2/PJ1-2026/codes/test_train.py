# An example of read in the data and train the model. The runner is implemented, while the model used for training need your implementation.
import mynn as nn
from draw_tools.plot import plot

import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

# fixed seed for experiment
np.random.seed(309)

train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        train_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)


# choose 10000 samples from train set as validation set.
idx = np.random.permutation(np.arange(num))
# save the index.
with open('idx.pickle', 'wb') as f:
        pickle.dump(idx, f)
train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

# normalize from [0, 255] to [0, 1]
train_imgs = train_imgs / train_imgs.max()
valid_imgs = valid_imgs / valid_imgs.max()

model_name = 'MLP'
dropout_rate = 0.0
save_dir = rf'./best_models/{model_name.lower()}'

if model_name == 'MLP':
        dropout_list = [dropout_rate] if dropout_rate > 0 else None
        model = nn.models.Model_MLP([train_imgs.shape[-1], 600, 10], 'ReLU', None, dropout_list)
elif model_name == 'CNN':
        model = nn.models.Model_CNN(conv_channels=8, dropout_rate=dropout_rate)
else:
        raise ValueError(f'Unknown model_name: {model_name}')

optimizer = nn.optimizer.SGD(init_lr=0.06, model=model)
scheduler = None
loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=train_labs.max()+1)

runner = nn.runner.RunnerM(model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler)

runner.train(
        [train_imgs, train_labs],
        [valid_imgs, valid_labs],
        num_epochs=5,
        log_iters=100,
        save_dir=save_dir,
)

_, axes = plt.subplots(1, 2)
axes.reshape(-1)
_.set_tight_layout(1)
plot(runner, axes)

plt.show()
