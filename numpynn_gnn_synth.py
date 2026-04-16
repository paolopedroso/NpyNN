"""
GCN on Synthetic Community Graph (10,000 nodes)
=============================================
Stochastic block model with 7 communities.
Includes standard GNN visualizations:
 - Training curves (loss + accuracy)
 - t-SNE of learned node embeddings colored by class
 - Subgraph visualization with predicted labels
"""
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import networkx as nx

from numpynn import *

_dir = os.path.dirname(os.path.abspath(__file__))
SAVE_PATH = os.path.join(_dir, "datasets/community_10k.npz")

################### generate or load dataset

def generate_dataset(
    num_nodes=10000,
    num_classes=7,
    num_features=32,
    p_intra=0.004,
    p_inter=0.0003,
    feature_noise=0.4,
    seed=42
):
    print("Generating graph...")
    np.random.seed(seed)

    nodes_per_class = num_nodes // num_classes
    remainder = num_nodes - nodes_per_class * num_classes
    y = np.concatenate([
        np.repeat(np.arange(num_classes), nodes_per_class),
        np.arange(remainder)
    ])
    num_nodes = len(y)

    # Node features: noisy class prototypes
    prototypes = np.random.randn(num_classes, num_features) * 2
    X = np.zeros((num_nodes, num_features))
    for c in range(num_classes):
        mask = y == c
        n = mask.sum()
        X[mask] = prototypes[c] + feature_noise * np.random.randn(n, num_features)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    # Adjacency: stochastic block model via vectorized numpy
    print("Building adjacency (this may take a moment)...")
    label_matrix = (y[:, None] == y[None, :])
    rand = np.random.rand(num_nodes, num_nodes)
    A = np.where(label_matrix, rand < p_intra, rand < p_inter).astype(np.float64)
    np.fill_diagonal(A, 0)
    A = np.maximum(A, A.T)  # symmetric

    # Train/test split (50/50 stratified)
    train_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)
    for c in range(num_classes):
        idx = np.where(y == c)[0]
        np.random.shuffle(idx)
        split = len(idx) // 2
        train_mask[idx[:split]] = True
        test_mask[idx[split:]] = True

    np.savez(SAVE_PATH, X=X, A=A, y=y,
             train_mask=train_mask, test_mask=test_mask)

    return X, A, y, train_mask, test_mask

# Load or generate
if os.path.exists(SAVE_PATH):
    print("Loading cached dataset...")
    data = np.load(SAVE_PATH)
    X, A, y = data["X"], data["A"], data["y"]
    train_mask, test_mask = data["train_mask"], data["test_mask"]
else:
    X, A, y, train_mask, test_mask = generate_dataset()

num_nodes = X.shape[0]
num_features = X.shape[1]
num_classes = len(np.unique(y))
num_edges = int(A.sum())

print(f"\nDataset: {num_nodes} nodes, {num_edges} edges, "
      f"{num_features} features, {num_classes} classes")
print(f"Train: {train_mask.sum()}, Test: {test_mask.sum()}")
print(f"Avg degree: {num_edges / num_nodes:.1f}\n")

################### build and train model
model = GraphModel(
    GCNLayer(num_features, 64),
    ReLU(),
    GCNLayer(64, num_classes),
    Softmax()
)
model.set(
    loss=CategoricalCrossEntropy(),
    optimizer=Adam(learning_rate=0.01),
    accuracy=Accuracy_Categorical()
)

print("Preprocessing graph...")
t0 = time.time()
model.set_graph(adj_matrix=A)
print(f"  set_graph: {time.time() - t0:.2f}s")

model.finalize()

print("\nTraining...\n")
t0 = time.time()
model.train(X, y, epochs=10, print_every=25)
train_time = time.time() - t0
print(f"\nTraining time: {train_time:.1f}s")

######################### evaluate

predictions = model.predict(X)
predicted_classes = np.argmax(predictions, axis=1)

train_acc = np.mean(predicted_classes[train_mask] == y[train_mask])
test_acc = np.mean(predicted_classes[test_mask] == y[test_mask])

print(f"\nResults:")
print(f"  Train accuracy: {train_acc:.3f}")
print(f"  Test accuracy:  {test_acc:.3f}")

######################### visualizations
# Extract learned embeddings from the last hidden layer (before softmax)
# Forward pass and grab output of ReLU (layer index 1)
_ = model.forward(X, training=False)
embeddings = model.layers[1].output  # ReLU output = hidden representations

colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
class_colors = colors[y]
pred_colors = colors[predicted_classes]

fig = plt.figure(figsize=(18, 12))

# Plot 1: Training Loss
ax1 = fig.add_subplot(2, 2, 1)
epochs = range(1, len(model.history['train_loss']) + 1)
ax1.plot(epochs, model.history['train_loss'], color='#e74c3c', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss')
ax1.grid(True, alpha=0.3)

# Plot 2: Training Accuracy
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(epochs, model.history['train_acc'], color='#2ecc71', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Training Accuracy')
ax2.grid(True, alpha=0.3)

# Plot 3: t-SNE of Learned Embeddings
print("\nComputing t-SNE (this takes a moment)...")
# Subsample for speed if dataset is large
max_tsne = 3000
if num_nodes > max_tsne:
    idx = np.random.choice(num_nodes, max_tsne, replace=False)
    tsne_X = embeddings[idx]
    tsne_y = y[idx]
    tsne_pred = predicted_classes[idx]
else:
    idx = np.arange(num_nodes)
    tsne_X = embeddings
    tsne_y = y
    tsne_pred = predicted_classes

tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
coords = tsne.fit_transform(tsne_X)

ax3 = fig.add_subplot(2, 2, 3)
for c in range(num_classes):
    mask = tsne_y == c
    ax3.scatter(coords[mask, 0], coords[mask, 1],
                c=[colors[c]], s=8, alpha=0.6, label=f'Class {c}')
ax3.set_title('t-SNE of Learned Embeddings (true labels)')
ax3.legend(markerscale=3, fontsize=8)
ax3.set_xticks([])
ax3.set_yticks([])

# Plot 4: Subgraph Visualization
ax4 = fig.add_subplot(2, 2, 4)

# Pick a seed node and extract its 2-hop neighborhood
seed = np.random.RandomState(42).choice(num_nodes)
neighbors_1 = set(np.where(A[seed] > 0)[0])
neighbors_2 = set()
for n in neighbors_1:
    neighbors_2.update(np.where(A[n] > 0)[0])
subgraph_nodes = sorted(list(neighbors_1 | neighbors_2 | {seed}))

# Cap at 200 nodes for readability
if len(subgraph_nodes) > 200:
    subgraph_nodes = subgraph_nodes[:200]

A_sub = A[np.ix_(subgraph_nodes, subgraph_nodes)]
G_sub = nx.from_numpy_array(A_sub)

node_colors_sub = [colors[predicted_classes[n]] for n in subgraph_nodes]
pos = nx.spring_layout(G_sub, seed=42, k=1.5/np.sqrt(len(subgraph_nodes)))

nx.draw_networkx_edges(G_sub, pos, alpha=0.15, ax=ax4, width=0.5)
nx.draw_networkx_nodes(G_sub, pos, node_color=node_colors_sub,
                       node_size=30, alpha=0.8, ax=ax4)
ax4.set_title(f'2-Hop Subgraph from Node {seed} (predicted labels)')
ax4.axis('off')

fig.suptitle(
    f'GCN on Community Graph, {num_nodes} nodes, {num_classes} classes, '
    f'Test Acc: {test_acc:.3f}',
    fontsize=14, fontweight='bold'
)
plt.tight_layout()
plt.savefig(os.path.join(_dir, 'plots/gcn_results.png'), dpi=150, bbox_inches='tight')
print(f"Saved gcn_results.png")
plt.show()