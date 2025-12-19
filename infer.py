import os
import argparse
import numpy as np
import pandas as pd
import torch

from .models import PromptGRN
from .data_utils import (
    load_expression_features,
    load_index_list,
    load_edge_triplets,
    build_adj_from_positive_edges,
    scipy_to_torch_sparse,
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--exp_file', type=str, required=True)
    p.add_argument('--tf_file', type=str, required=True)
    p.add_argument('--model_ckpt', type=str, required=True)
    p.add_argument('--out_csv', type=str, required=True)
    p.add_argument('--train_file', type=str, default=None,
                   help='Optional: provide Train_set.csv to rebuild initial graph; otherwise uses empty graph.')
    p.add_argument('--topk', type=int, default=200000, help='Number of top edges to export.')
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x = load_expression_features(args.exp_file, normalize=True)
    x_expr = torch.from_numpy(x).to(device)
    num_genes = x.shape[0]
    gene_ids = torch.arange(num_genes, device=device)

    tf_indices = load_index_list(args.tf_file, col='index').astype(np.int64)

    # Build initial graph if train file is provided; else use identity (no edges)
    if args.train_file is not None:
        train_arr = load_edge_triplets(args.train_file)
        adj_sp = build_adj_from_positive_edges(train_arr, num_nodes=num_genes, directed=True, self_loop=False)
    else:
        adj_sp = scipy.sparse.identity(num_genes, dtype=np.float32).todok()  # fallback
    adj = scipy_to_torch_sparse(adj_sp).to(device)

    # Instantiate model with default hyperparams (must match training for best results)
    model = PromptGRN(num_genes=num_genes, x_expr_dim=x_expr.shape[1]).to(device)
    state = torch.load(args.model_ckpt, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    # Score all TF -> gene pairs (excluding self)
    tf_list = tf_indices.tolist()
    pairs = []
    for tf in tf_list:
        for tg in range(num_genes):
            if tf == tg:
                continue
            pairs.append((tf, tg))
    pairs = np.array(pairs, dtype=np.int64)

    batch = 200000  # adjust for memory
    scores = np.zeros((pairs.shape[0],), dtype=np.float32)

    with torch.no_grad():
        # Precompute node embeddings once
        h_final = model.encode(x_expr, adj, gene_ids)  # [N,d]
        W = model.link_pred.W  # bilinear weights

        for start in range(0, pairs.shape[0], batch):
            end = min(start + batch, pairs.shape[0])
            p = torch.from_numpy(pairs[start:end]).to(device)
            hi = h_final[p[:, 0]]
            hj = h_final[p[:, 1]]
            logits = torch.sum((hi @ W) * hj, dim=1)
            prob = torch.sigmoid(logits).cpu().numpy()
            scores[start:end] = prob

    # Top-K export
    topk = min(args.topk, scores.shape[0])
    idx = np.argpartition(-scores, topk - 1)[:topk]
    idx = idx[np.argsort(-scores[idx])]

    out = pd.DataFrame({
        "TF": pairs[idx, 0],
        "Target": pairs[idx, 1],
        "Score": scores[idx]
    })
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Saved top-{topk} edges to {args.out_csv}")

if __name__ == '__main__':
    main()
