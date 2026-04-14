import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

fashion_mnist_labels = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # normalize between -1 and 1
])

train_dataset = datasets.FashionMNIST(root='datasets', train=True,
                                       download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='datasets', train=False,
                                       download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset,  batch_size=128, shuffle=False)

neurons = 128

############################ Initialize model

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

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters()
                       , lr=0.001
                       , eps=1e-7
                       , betas=(0.9, 0.999))

# inverse time LR decay per step to match numpynn Adam decay=5e-5
decay = 5e-5
scheduler = optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=lambda step: 1.0 / (1.0 + decay * step)
)

############################ Train

history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': [],
    'lr': [],
}

for epoch in range(1, 6):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    for step, (X_batch, y_batch) in enumerate(train_loader):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        output = model(X_batch)
        loss = loss_fn(output, y_batch)
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * y_batch.size(0)
        running_correct += (output.argmax(dim=1) == y_batch).sum().item()
        running_total += y_batch.size(0)

        if not step % 100 or step == len(train_loader) - 1:
            acc = (output.argmax(dim=1) == y_batch).float().mean()
            print(f'epoch: {epoch}, step: {step}, '
                  f'acc: {acc:.3f}, loss: {loss:.3f}')

    train_loss = running_loss / running_total
    train_acc = running_correct / running_total

    model.eval()
    with torch.no_grad():
        v_correct = 0
        v_total = 0
        v_loss = 0.0
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            v_loss += loss_fn(output, y_batch).item() * y_batch.size(0)
            v_correct += (output.argmax(dim=1) == y_batch).sum().item()
            v_total += y_batch.size(0)
    val_loss = v_loss / v_total
    val_acc = v_correct / v_total

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['lr'].append(optimizer.param_groups[0]['lr'])

    print(f'epoch: {epoch}, '
          f'train_acc: {train_acc:.3f}, train_loss: {train_loss:.3f}, '
          f'val_acc: {val_acc:.3f}, val_loss: {val_loss:.3f}')

############################ Evaluate

model.eval()
with torch.no_grad():
    correct = total = val_loss = 0
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        output = model(X_batch)
        val_loss += loss_fn(output, y_batch).item()
        correct += (output.argmax(dim=1) == y_batch).sum().item()
        total += y_batch.size(0)

    print(f'validation, acc: {correct/total:.3f}, '
          f'loss: {val_loss/len(test_loader):.3f}')

# save model
if os.environ("SAVE") == 1:
    os.makedirs('data_history', exist_ok=True)
    torch.save(model.state_dict(), 'models/torch_fashion_mnist.pth')
    np.savez('data_history/torch_fashion_mnist_history.npz',
            train_loss=np.array(history['train_loss']),
            train_acc=np.array(history['train_acc']),
            val_loss=np.array(history['val_loss']),
            val_acc=np.array(history['val_acc']),
            lr=np.array(history['lr']))
