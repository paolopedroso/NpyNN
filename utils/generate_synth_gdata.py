import numpy as np

def generate_gdataset(
    num_nodes=80_000,
    num_classes=7,
    num_features=32,
    p_intra=0.0015,
    p_inter=0.001,
    feature_noise=2.0,
    seed=42
):
    """
    ### `generate_gdataset`

    Generates synthetic dataset for GNN models.

    Args:
        num_nodes (int): Number of nodes. Default = 80_000
        num_classes (int): Number of classes. Default = 7
        num_features (int): Number of features. Default = 32
        p_intra (float):  Default = 0.0015
        p_inter (float):  Default = 0.001
        feature_noise (float):  Default = 2.0
        seed (int): Seed. Default = 42
    """
    print("Generating graph...")
    np.random.seed(seed)

    nodes_per_class = num_nodes // num_classes
    remainder = num_nodes - nodes_per_class * num_classes
    y = np.concatenate([
        np.repeat(np.arange(num_classes), nodes_per_class),
        np.arange(remainder)
    ])
    num_nodes = len(y)

    # noisy class prototypes
    prototypes = np.random.randn(num_classes, num_features) * 2
    X = np.zeros((num_nodes, num_features))
    for c in range(num_classes):
        mask = y == c
        n = mask.sum()
        X[mask] = prototypes[c] + feature_noise * np.random.randn(n, num_features)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    # Sparse edge sampling (2, E)
    src_list, dst_list = [], []

    # sample within each class
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

    train_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)
    for c in range(num_classes):
        idx = np.where(y == c)[0]
        np.random.shuffle(idx)
        split = len(idx) // 2
        train_mask[idx[:split]] = True
        test_mask[idx[split:]] = True

    return X, edge_index, y, train_mask, test_mask