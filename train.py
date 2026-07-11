import os
import argparse
import numpy as np
import torch

from .models import PromptGRN
from .data_utils import (
    load_expression_features,
    load_index_list,
    load_edge_triplets,
    build_adj_from_positive_edges,
    scipy_to_torch_sparse,
    make_dynamic_train_pairs,
)
from .metrics import evaluate_binary

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--exp_file', type=str, required=True, help='Expression feature CSV (rows=genes).')
    p.add_argument('--tf_file', type=str, required=True, help='TF list CSV with column index.')
    p.add_argument('--train_file', type=str, required=True, help='Train_set.csv with TF,Target,Label.')
    p.add_argument('--val_file', type=str, required=True, help='Validation_set.csv with TF,Target,Label.')
    p.add_argument('--save_dir', type=str, default='./model', help='Directory to save checkpoints.')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=3e-3)
    p.add_argument('--seed', type=int, default=8)
    p.add_argument('--neg_ratio', type=int, default=1, help='Negative sampling ratio per epoch.')
    p.add_argument('--lambda_dgi', type=float, default=0.5)
    # model hyperparams
    p.add_argument('--d_prompt', type=int, default=128)
    p.add_argument('--gnn_h1', type=int, default=64)
    p.add_argument('--gnn_h2', type=int, default=64)
    p.add_argument('--gnn_head1', type=int, default=3)
    p.add_argument('--gnn_head2', type=int, default=3)
    p.add_argument('--reduction', type=str, default='concate', choices=['mean', 'concate'])
    p.add_argument('--trans_d_model', type=int, default=128)
    p.add_argument('--trans_nhead', type=int, default=8)
    p.add_argument('--trans_layers', type=int, default=2)
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # Reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    x = load_expression_features(args.exp_file, normalize=True)  # [N,D]
    x_expr = torch.from_numpy(x).to(device)

    tf_indices = load_index_list(args.tf_file, col='index').tolist()
    train_arr = load_edge_triplets(args.train_file)  # [M,3]
    val_arr = load_edge_triplets(args.val_file)

    num_genes = x.shape[0]
    gene_ids = torch.arange(num_genes, device=device)

    # Initial regulatory graph (ONLY positive edges)
    adj_sp = build_adj_from_positive_edges(train_arr, num_nodes=num_genes, directed=True, self_loop=False)
    adj = scipy_to_torch_sparse(adj_sp).to(device)

    # Validation pairs/labels (use the file as-is)
    val_pairs = torch.from_numpy(val_arr[:, :2].astype(np.int64)).to(device)
    val_labels = torch.from_numpy(val_arr[:, -1].astype(np.float32)).to(device)

    # Model
    model = PromptGRN(
        num_genes=num_genes,
        x_expr_dim=x_expr.shape[1],
        d_prompt=args.d_prompt,
        gnn_h1=args.gnn_h1,
        gnn_h2=args.gnn_h2,
        gnn_head1=args.gnn_head1,
        gnn_head2=args.gnn_head2,
        reduction=args.reduction,
        trans_d_model=args.trans_d_model,
        trans_nhead=args.trans_nhead,
        trans_layers=args.trans_layers,
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_aupr = -1.0
    best_path = os.path.join(args.save_dir, 'PromptGRN.pt')

    for epoch in range(1, args.epochs + 1):
        model.train()

        # Dynamic negative sampling each epoch
        pairs_np, labels_np = make_dynamic_train_pairs(
            train_arr, num_nodes=num_genes, tf_indices=tf_indices, neg_ratio=args.neg_ratio, seed=epoch
        )
        train_pairs = torch.from_numpy(pairs_np).long().to(device)
        train_labels = torch.from_numpy(labels_np).float().to(device)

        loss, loss_grn, loss_dgi = model.compute_losses(
            x_expr=x_expr,
            adj_reg=adj,
            pairs=train_pairs,
            labels=train_labels,
            gene_ids=gene_ids,
            lambda_dgi=args.lambda_dgi
        )

        optim.zero_grad()
        loss.backward()
        optim.step()

        # Validation
        model.eval()
        with torch.no_grad():
            logits, _ = model(x_expr, adj, val_pairs.long(), gene_ids)
            prob = torch.sigmoid(logits).detach().cpu().numpy()
            auc, aupr, aupr_norm = evaluate_binary(val_labels.detach().cpu().numpy(), prob)

        if aupr > best_aupr:
            best_aupr = aupr
            torch.save(model.state_dict(), best_path)

        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | loss={loss.item():.4f} (grn={loss_grn.item():.4f}, dgi={loss_dgi.item():.4f}) "
                  f"| AUC={auc:.4f} AUPR={aupr:.4f} AUPR_norm={aupr_norm:.4f}")

    print(f"Done. Best checkpoint saved to: {best_path}")

if __name__ == '__main__':
    main()
