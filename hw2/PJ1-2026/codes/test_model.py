import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

model_name = 'MLP'
model_path = rf'.\best_models\{model_name.lower()}\best_model.pickle'
if model_name == 'MLP':
        model = nn.models.Model_MLP()
elif model_name == 'CNN':
        model = nn.models.Model_CNN()
else:
        raise ValueError(f'Unknown model_name: {model_name}')
model.load_model(model_path)
if hasattr(model, 'eval'):
        model.eval()

test_images_path = r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz'
test_labels_path = r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz'

with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        test_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)

test_imgs = test_imgs / test_imgs.max()

logits = model(test_imgs)
print(nn.metric.accuracy(logits, test_labs))
