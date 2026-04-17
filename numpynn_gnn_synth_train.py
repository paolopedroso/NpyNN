import numpy as np
import os
import time
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import networkx as nx

from numpynn import *

from utils.generate_synth_gdata import generate_gdataset

_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(_dir, "datasets"), exist_ok=True)
os.makedirs(os.path.join(_dir, "plots"), exist_ok=True)

################### generate or load dataset

X, edge_index, y, train_mask, test_mask = generate_gdataset()

num_nodes = X.shape[0]
num_features = X.shape[1]
num_classes = len(np.unique(y))
num_edges = edge_index.shape[1]

print(f"\nDataset: {num_nodes} nodes, {num_edges} edges, "
      f"{num_features} features, {num_classes} classes")
print(f"Train: {train_mask.sum()}, Test: {test_mask.sum()}")
print(f"Avg degree: {num_edges / num_nodes:.1f}\n")

################### build and train model

MODEL_PATH = "cache/models/numpynn_synth_80k_82p__gdata.model"

if os.path.exists(MODEL_PATH):
    print(f"Loading saved model from {MODEL_PATH}...")
    model = Model.load(MODEL_PATH)
    model.set_graph(edge_index=edge_index)  # re-attach graph after load
else:
    print("No saved model found, training from scratch...")
    model = GraphModel(
        GCNLayer(num_features, 128),
        ReLU(),
        GCNLayer(128, num_classes),
        Softmax()
    )
    model.set(
        loss=CategoricalCrossEntropy(),
        optimizer=Adam(
            learning_rate=0.01,
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

    model.save(MODEL_PATH)

######################### evaluate
predictions = model.predict(X)
predicted_classes = np.argmax(predictions, axis=1)

train_acc = np.mean(predicted_classes[train_mask] == y[train_mask])
test_acc = np.mean(predicted_classes[test_mask] == y[test_mask])

print(f"\nResults:")
print(f"  Train accuracy: {train_acc:.3f}")
print(f"  Test accuracy:  {test_acc:.3f}")

model.save("models/numpynn_synth_80k_82p__gdata.model")

######################### visualizations
# Extract learned embeddings from the last hidden layer (before softmax)
_ = model.forward(X, training=False)
embeddings = model.layers[1].output  # ReLU output = hidden representations

colors = plt.cm.tab10(np.linspace(0, 1, num_classes))

fig = plt.figure(figsize=(18, 12))

######################### Training Loss
ax1 = fig.add_subplot(2, 2, 1)
epochs = range(1, len(model.history['train_loss']) + 1)
ax1.plot(epochs, model.history['train_loss'], color='#e74c3c', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss')
ax1.grid(True, alpha=0.3)

######################### Training Accuracy
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(epochs, model.history['train_acc'], color='#2ecc71', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Training Accuracy')
ax2.grid(True, alpha=0.3)

######################### t-SNE of Learned Embeddings
print("\nComputing t-SNE (this takes a moment)...")
max_tsne = 3000
if num_nodes > max_tsne:
    idx = np.random.choice(num_nodes, max_tsne, replace=False)
    tsne_X = embeddings[idx]
    tsne_y = y[idx]
else:
    tsne_X = embeddings
    tsne_y = y

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

######################### Subgraph Visualization (sparse-friendly)
ax4 = fig.add_subplot(2, 2, 4)

# Build a dict mapping source -> targets once, then do 2-hop BFS
print("Building subgraph neighborhood...")
src_arr, dst_arr = edge_index[0], edge_index[1]

# Pick a seed node
seed = int(np.random.RandomState(42).choice(num_nodes))

# 1-hop: all targets where source == seed
neighbors_1 = set(dst_arr[src_arr == seed].tolist())

# 2-hop: neighbors of neighbors
neighbors_2 = set()
for n in neighbors_1:
    neighbors_2.update(dst_arr[src_arr == n].tolist())

subgraph_nodes = sorted(list(neighbors_1 | neighbors_2 | {seed}))

# Cap at 200 nodes for readability
if len(subgraph_nodes) > 200:
    subgraph_nodes = subgraph_nodes[:200]

# Build small dense adjacency just for this subgraph (fine — it's tiny)
subgraph_set = set(subgraph_nodes)
node_to_idx = {n: i for i, n in enumerate(subgraph_nodes)}
n_sub = len(subgraph_nodes)
A_sub = np.zeros((n_sub, n_sub))

# Filter edges where both endpoints are in the subgraph
mask = np.isin(src_arr, subgraph_nodes) & np.isin(dst_arr, subgraph_nodes)
for s, d in zip(src_arr[mask], dst_arr[mask]):
    A_sub[node_to_idx[s], node_to_idx[d]] = 1

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
plt.savefig(os.path.join(_dir, 'plots/gcn_synth_80k_results.png'), dpi=150, bbox_inches='tight')
print(f"Saved gcn_synth_80k_results.png")
plt.show()