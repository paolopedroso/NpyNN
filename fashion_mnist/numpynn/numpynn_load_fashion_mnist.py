import os
import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from numpynn import *

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.FashionMNIST(root='datasets', train=True,  download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='datasets', train=False, download=True, transform=transform)

X = train_dataset.data.numpy().astype(np.float32)
y = train_dataset.targets.numpy()
X_test = test_dataset.data.numpy().astype(np.float32)
y_test = test_dataset.targets.numpy()

# shuffle training data
keys = np.random.permutation(X.shape[0])
X = X[keys]
y = y[keys]

X = (X.reshape(X.shape[0], -1) - 127.5) / 127.5
X_test = (X_test.reshape(X_test.shape[0], -1) - 127.5) / 127.5

model = Model.load('cache/models/numpynn_fashion_mnist.model')

############################ Evaluate the model

model.evaluate(X_test, y_test)

############################ Training history

history_path = 'cache/data_history/numpynn_fashion_mnist_history.npz'
if os.path.exists(history_path):
    h = np.load(history_path)
    epochs = np.arange(1, len(h['train_loss']) + 1)

    fig_h, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, h['train_loss'], marker='o', label='train')
    axes[0].plot(epochs, h['val_loss'], marker='o', label='val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, h['train_acc'], marker='o', label='train')
    axes[1].plot(epochs, h['val_acc'], marker='o', label='val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(epochs, h['lr'], marker='o', color='tab:green')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning rate')
    axes[2].set_title('Learning rate')
    axes[2].grid(True, alpha=0.3)

    fig_h.suptitle('NumpyNN - Fashion MNIST Training History')
    fig_h.tight_layout()

    os.makedirs('plots', exist_ok=True)
    history_out = os.path.join('plots', 'numpynn_fashion_mnist_history.png')
    fig_h.savefig(history_out, dpi=150, bbox_inches='tight')
    print(f'Saved training history to {history_out}')
else:
    print(f'No history file at {history_path}, skipping history plot')

############################ Confusion matrix
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
num_classes = len(class_names)

probs = model.predict(X_test, batch_size=128)
y_pred = np.argmax(probs, axis=1)

cm = np.zeros((num_classes, num_classes), dtype=np.int64)
for true, pred in zip(y_test, y_pred):
    cm[true, pred] += 1

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm, cmap='Blues')
fig.colorbar(im, ax=ax)

ax.set_xticks(range(num_classes))
ax.set_yticks(range(num_classes))
ax.set_xticklabels(class_names, rotation=45, ha='right')
ax.set_yticklabels(class_names)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('NumpyNN - Fashion MNIST Confusion Matrix')

thresh = cm.max() / 2
for i in range(num_classes):
    for j in range(num_classes):
        ax.text(j, i, cm[i, j], ha='center', va='center',
                color='white' if cm[i, j] > thresh else 'black')

plt.tight_layout()

plots_dir = 'plots'
os.makedirs(plots_dir, exist_ok=True)
output_path = os.path.join(plots_dir, 'numpynn_fashion_mnist_confusion_matrix.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved confusion matrix to {output_path}")
plt.show()