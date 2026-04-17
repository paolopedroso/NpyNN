"""
PyTorch Geometric GCN on Synthetic Community Graph
=============================================
Mirrors numpynn_gnn_synth.py exactly — same SBM generator, same seed,
same architecture, same hyperparameters — for direct comparison.
"""
import numpy as np
import os
import time
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

_dir = os.path.dirname(os.path.abspath(__file__))

from utils.generate_synth_gdata import generate_gdataset

################### generate dataset (identical to numpynn version)

X_np, edge_index_np, y_np, train_mask_np, test_mask_np = generate_gdataset()

# Convert to torch tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X = torch.tensor(X_np, dtype=torch.float32, device=device)
edge_index = torch.tensor(edge_index_np, dtype=torch.long, device=device)
y = torch.tensor(y_np, dtype=torch.long, device=device)
train_mask = torch.tensor(train_mask_np, dtype=torch.bool, device=device)
test_mask = torch.tensor(test_mask_np, dtype=torch.bool, device=device)

num_nodes = X.shape[0]
num_features = X.shape[1]
num_classes = int(y.max().item()) + 1
num_edges = edge_index.shape[1]

print(f"\nDataset: {num_nodes} nodes, {num_edges} edges, "
      f"{num_features} features, {num_classes} classes")
print(f"Train: {train_mask.sum().item()}, Test: {test_mask.sum().item()}")
print(f"Avg degree: {num_edges / num_nodes:.1f}")
print(f"Device: {device}\n")

################### build model — matches numpynn architecture

class GCN(torch.nn.Module):
    def __init__(self, in_features, hidden, out_features):
        super().__init__()
        self.conv1 = GCNConv(in_features, hidden)
        self.conv2 = GCNConv(hidden, out_features)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x  # logits; F.cross_entropy applies log_softmax internally


# Match numpynn hyperparameters exactly
HIDDEN = 128
LR = 0.01
WEIGHT_DECAY = 1e-5
EPOCHS = 30

torch.manual_seed(42)
model = GCN(num_features, HIDDEN, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

print("Training...\n")
t0 = time.time()
for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()
    out = model(X, edge_index)
    loss = F.cross_entropy(out, y)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        pred = out.argmax(dim=1)
        train_acc = (pred == y).float().mean().item()

    print(f'epoch: {epoch}, loss: {loss.item():.3f}, acc: {train_acc:.3f}')

train_time = time.time() - t0
print(f"\nTraining time: {train_time:.1f}s")

################### evaluate
model.eval()
with torch.no_grad():
    out = model(X, edge_index)
    pred = out.argmax(dim=1)

train_acc = (pred[train_mask] == y[train_mask]).float().mean().item()
test_acc = (pred[test_mask] == y[test_mask]).float().mean().item()

print(f"\nResults:")
print(f"  Train accuracy: {train_acc:.3f}")
print(f"  Test accuracy:  {test_acc:.3f}")
