import os
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
import torch

def load_expression_features(exp_csv_path, normalize=True):
    df = pd.read_csv(exp_csv_path, index_col=0)
    x = df.values.astype(np.float32)
    if normalize:
        # Standardize features (columns) across genes (rows)
        scaler = StandardScaler()
        x = scaler.fit_transform(x).astype(np.float32)
    return x

def load_index_list(csv_path, col="index"):
    df = pd.read_csv(csv_path, index_col=0) if "Unnamed: 0" in open(csv_path, 'r', encoding='utf-8', errors='ignore').readline() else pd.read_csv(csv_path)
    if col not in df.columns:
        # fallback: if the file has only one column
        if df.shape[1] == 1:
            return df.iloc[:, 0].values.astype(np.int64)
        raise ValueError(f"Column '{col}' not found in {csv_path}. Available: {list(df.columns)}")
    return df[col].values.astype(np.int64)

def load_edge_triplets(csv_path):
    df = pd.read_csv(csv_path)
    # Support either with or without header index column
    cols = df.columns.tolist()
    if "TF" in cols and "Target" in cols and "Label" in cols:
        arr = df[["TF", "Target", "Label"]].values
    else:
        arr = df.values
    return arr.astype(np.float32)

def build_adj_from_positive_edges(edge_triplets, num_nodes, directed=True, self_loop=False):
    adj = sp.dok_matrix((num_nodes, num_nodes), dtype=np.float32)
    for tf, tg, y in edge_triplets:
        if int(y) != 1:
            continue
        tf = int(tf); tg = int(tg)
        adj[tf, tg] = 1.0
        if not directed:
            adj[tg, tf] = 1.0
    if self_loop:
        adj = adj + sp.identity(num_nodes, dtype=np.float32)
    return adj.todok()

def scipy_to_torch_sparse(adj):
    """Convert scipy COO/DOK/CSR to torch sparse COO tensor."""
    coo = adj.tocoo()
    idx = torch.LongTensor([coo.row, coo.col])
    val = torch.from_numpy(coo.data).float()
    return torch.sparse_coo_tensor(idx, val, coo.shape).coalesce()

def build_positive_set(edge_triplets):
    """Extract positive edges and a python set for membership checks."""
    pos = edge_triplets[edge_triplets[:, -1] == 1][:, :2].astype(np.int64)
    pos_set = set((int(i), int(j)) for i, j in pos)
    return pos, pos_set

def negative_sampling(num_nodes, pos_set, num_samples, tf_indices=None, seed=None):
    if seed is not None:
        random.seed(seed)

    neg = []
    while len(neg) < num_samples:
        tf = random.choice(tf_indices) if tf_indices is not None else random.randrange(num_nodes)
        tg = random.randrange(num_nodes)
        if tf == tg:
            continue
        if (tf, tg) in pos_set:
            continue
        neg.append((tf, tg))
    return np.array(neg, dtype=np.int64)

def make_dynamic_train_pairs(edge_triplets, num_nodes, tf_indices=None, neg_ratio=1, seed=None):
    pos_edges, pos_set = build_positive_set(edge_triplets)
    num_pos = pos_edges.shape[0]
    neg_edges = negative_sampling(num_nodes, pos_set, num_pos * neg_ratio, tf_indices=tf_indices, seed=seed)

    pairs = np.vstack([pos_edges, neg_edges])
    labels = np.hstack([np.ones(num_pos), np.zeros(len(neg_edges))]).astype(np.float32)

    perm = np.random.permutation(len(labels))
    return pairs[perm], labels[perm]
