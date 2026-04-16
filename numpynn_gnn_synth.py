"""
GCN on Synthetic Community Graph
=============================================
Stochastic block model with 7 communities.
Sparse edge generation — scales to 100k+ nodes without 74GB allocations.

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
os.makedirs(os.path.join(_dir, "datasets"), exist_ok=True)
os.makedirs(os.path.join(_dir, "plots"), exist_ok=True)

################### generate or load dataset

def generate_dataset(
    num_nodes=80_000,
    num_classes=7,
    num_features=32,
    p_intra=0.0015,
    p_inter=0.001,
    feature_noise=2.0,
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

    # Sparse edge sampling: never materialize N x N matrix
    print("Building edges (sparse)...")
    src_list, dst_list = [], []

    # Intra-class edges: sample within each class
    for c in range(num_classes):
        nodes_c = np.where(y == c)[0]
        n_c = len(nodes_c)
        expected = p_intra * n_c * (n_c - 1) / 2
        n_edges = np.random.poisson(expected)
        s = np.random.choice(nodes_c, n_edges)
        d = np.random.choice(nodes_c, n_edges)
        keep = s != d
        src_list.append(s[keep])
        dst_list.append(d[keep])

    # Inter-class edges: sample globally, reject same-class pairs
    expected_inter = p_inter * num_nodes * (num_nodes - 1) / 2
    n_edges = np.random.poisson(expected_inter)
    s = np.random.randint(0, num_nodes, n_edges)
    d = np.random.randint(0, num_nodes, n_edges)
    keep = (y[s] != y[d]) & (s != d)
    src_list.append(s[keep])
    dst_list.append(d[keep])

    src = np.concatenate(src_list)
    dst = np.concatenate(dst_list)

    # Symmetrize: (i,j) implies (j,i)
    all_src = np.concatenate([src, dst])
    all_dst = np.concatenate([dst, src])

    # Deduplicate
    edges = np.unique(np.stack([all_src, all_dst], axis=1), axis=0)
    edge_index = edges.T.astype(np.int64)  # shape (2, E)

    # Train/test split (50/50 stratified)
    train_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)
    for c in range(num_classes):
        idx = np.where(y == c)[0]
        np.random.shuffle(idx)
        split = len(idx) // 2
        train_mask[idx[:split]] = True
        test_mask[idx[split:]] = True

    return X, edge_index, y, train_mask, test_mask


X, edge_index, y, train_mask, test_mask = generate_dataset()

num_nodes = X.shape[0]
num_features = X.shape[1]
num_classes = len(np.unique(y))
num_edges = edge_index.shape[1]

print(f"\nDataset: {num_nodes} nodes, {num_edges} edges, "
      f"{num_features} features, {num_classes} classes")
print(f"Train: {train_mask.sum()}, Test: {test_mask.sum()}")
print(f"Avg degree: {num_edges / num_nodes:.1f}\n")

################### build and train model
model = GraphModel(
    GCNLayer(num_features, 128),
    ReLU(),
    GCNLayer(128, num_classes),
    Softmax()
)
model.set(
    loss=CategoricalCrossEntropy(),
    optimizer=Adam(
        learning_rate=0.001,
        decay=1e-5
    ),
    accuracy=Accuracy_Categorical()
)

print("Preprocessing graph...")
t0 = time.time()
model.set_graph(edge_index=edge_index)
print(f"  set_graph: {time.time() - t0:.2f}s")

model.finalize()

print("\nTraining...\n")
t0 = time.time()
model.train(X, y, epochs=30, print_every=25)
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

# ######################### visualizations
# # Extract learned embeddings from the last hidden layer (before softmax)
# _ = model.forward(X, training=False)
# embeddings = model.layers[1].output  # ReLU output = hidden representations

# colors = plt.cm.tab10(np.linspace(0, 1, num_classes))

# fig = plt.figure(figsize=(18, 12))

# # Plot 1: Training Loss
# ax1 = fig.add_subplot(2, 2, 1)
# epochs = range(1, len(model.history['train_loss']) + 1)
# ax1.plot(epochs, model.history['train_loss'], color='#e74c3c', linewidth=2)
# ax1.set_xlabel('Epoch')
# ax1.set_ylabel('Loss')
# ax1.set_title('Training Loss')
# ax1.grid(True, alpha=0.3)

# # Plot 2: Training Accuracy
# ax2 = fig.add_subplot(2, 2, 2)
# ax2.plot(epochs, model.history['train_acc'], color='#2ecc71', linewidth=2)
# ax2.set_xlabel('Epoch')
# ax2.set_ylabel('Accuracy')
# ax2.set_title('Training Accuracy')
# ax2.grid(True, alpha=0.3)

# # Plot 3: t-SNE of Learned Embeddings
# print("\nComputing t-SNE (this takes a moment)...")
# max_tsne = 3000
# if num_nodes > max_tsne:
#     idx = np.random.choice(num_nodes, max_tsne, replace=False)
#     tsne_X = embeddings[idx]
#     tsne_y = y[idx]
# else:
#     tsne_X = embeddings
#     tsne_y = y

# tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
# coords = tsne.fit_transform(tsne_X)

# ax3 = fig.add_subplot(2, 2, 3)
# for c in range(num_classes):
#     mask = tsne_y == c
#     ax3.scatter(coords[mask, 0], coords[mask, 1],
#                 c=[colors[c]], s=8, alpha=0.6, label=f'Class {c}')
# ax3.set_title('t-SNE of Learned Embeddings (true labels)')
# ax3.legend(markerscale=3, fontsize=8)
# ax3.set_xticks([])
# ax3.set_yticks([])

# # Plot 4: Subgraph Visualization (sparse-friendly)
# ax4 = fig.add_subplot(2, 2, 4)

# # Build a dict mapping source -> targets once, then do 2-hop BFS
# print("Building subgraph neighborhood...")
# src_arr, dst_arr = edge_index[0], edge_index[1]

# # Pick a seed node
# seed = int(np.random.RandomState(42).choice(num_nodes))

# # 1-hop: all targets where source == seed
# neighbors_1 = set(dst_arr[src_arr == seed].tolist())

# # 2-hop: neighbors of neighbors
# neighbors_2 = set()
# for n in neighbors_1:
#     neighbors_2.update(dst_arr[src_arr == n].tolist())

# subgraph_nodes = sorted(list(neighbors_1 | neighbors_2 | {seed}))

# # Cap at 200 nodes for readability
# if len(subgraph_nodes) > 200:
#     subgraph_nodes = subgraph_nodes[:200]

# # Build small dense adjacency just for this subgraph (fine — it's tiny)
# subgraph_set = set(subgraph_nodes)
# node_to_idx = {n: i for i, n in enumerate(subgraph_nodes)}
# n_sub = len(subgraph_nodes)
# A_sub = np.zeros((n_sub, n_sub))

# # Filter edges where both endpoints are in the subgraph
# mask = np.isin(src_arr, subgraph_nodes) & np.isin(dst_arr, subgraph_nodes)
# for s, d in zip(src_arr[mask], dst_arr[mask]):
#     A_sub[node_to_idx[s], node_to_idx[d]] = 1

# G_sub = nx.from_numpy_array(A_sub)

# node_colors_sub = [colors[predicted_classes[n]] for n in subgraph_nodes]
# pos = nx.spring_layout(G_sub, seed=42, k=1.5/np.sqrt(len(subgraph_nodes)))

# nx.draw_networkx_edges(G_sub, pos, alpha=0.15, ax=ax4, width=0.5)
# nx.draw_networkx_nodes(G_sub, pos, node_color=node_colors_sub,
#                        node_size=30, alpha=0.8, ax=ax4)
# ax4.set_title(f'2-Hop Subgraph from Node {seed} (predicted labels)')
# ax4.axis('off')

# fig.suptitle(
#     f'GCN on Community Graph, {num_nodes} nodes, {num_classes} classes, '
#     f'Test Acc: {test_acc:.3f}',
#     fontsize=14, fontweight='bold'
# )
# plt.tight_layout()
# plt.savefig(os.path.join(_dir, 'plots/gcn_results.png'), dpi=150, bbox_inches='tight')
# print(f"Saved gcn_results.png")
# plt.show()