import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import datasets, transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_dataset = datasets.FashionMNIST(root='datasets', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

neurons = 128

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, neurons),
            nn.ReLU(),
            nn.Linear(neurons, neurons),
            nn.ReLU(),
            nn.Linear(neurons, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.network(x)

model = Model().to(device)
model.load_state_dict(torch.load('cache/models/torch_fashion_mnist.pth', map_location=device))

############################ Evaluate the model

model.eval()
with torch.no_grad():
    correct = total = val_loss = 0
    loss_fn = nn.CrossEntropyLoss()
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        output = model(X_batch)
        val_loss += loss_fn(output, y_batch).item()
        correct += (output.argmax(dim=1) == y_batch).sum().item()
        total += y_batch.size(0)

    print(f'validation, acc: {correct/total:.3f}, '
          f'loss: {val_loss/len(test_loader):.3f}')

############################ Training history

history_path = 'cache/data_history/torch_fashion_mnist_history.npz'
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

    fig_h.suptitle('PyTorch - Fashion MNIST Training History')
    fig_h.tight_layout()

    os.makedirs('plots', exist_ok=True)
    history_out = os.path.join('plots', 'torch_fashion_mnist_history.png')
    fig_h.savefig(history_out, dpi=150, bbox_inches='tight')
    print(f'Saved training history to {history_out}')
else:
    print(f'No history file at {history_path}, skipping history plot')

############################ Confusion matrix
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
num_classes = len(class_names)

all_preds, all_targets = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        output  = model(X_batch)
        all_preds.extend(output.argmax(dim=1).cpu().numpy())
        all_targets.extend(y_batch.numpy())

y_pred  = np.array(all_preds)
y_test  = np.array(all_targets)

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
ax.set_title('PyTorch - Fashion MNIST Confusion Matrix')

thresh = cm.max() / 2
for i in range(num_classes):
    for j in range(num_classes):
        ax.text(j, i, cm[i, j], ha='center', va='center',
                color='white' if cm[i, j] > thresh else 'black')

plt.tight_layout()

plots_dir = 'plots'
os.makedirs(plots_dir, exist_ok=True)
output_path = os.path.join(plots_dir, 'torch_fashion_mnist_confusion_matrix.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved confusion matrix to {output_path}")
plt.show()
