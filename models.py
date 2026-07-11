import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================
# AttentionLayer (GAT-style) - adapted from your original code
# =========================================================
class AttentionLayer(nn.Module):
    """Dense attention over adjacency mask.

    Note:
        This uses adj.to_dense() and builds an NxN attention matrix.
        For large N, replace with sparse attention.
    """
    def __init__(self, input_dim, output_dim, alpha=0.2, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha

        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.weight_interact = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.a = nn.Parameter(torch.zeros(size=(2 * output_dim, 1)))

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.weight_interact.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        if self.bias is not None:
            self.bias.data.fill_(0)

    def _prepare_attentional_mechanism_input(self, x):
        Wh1 = torch.matmul(x, self.a[:self.output_dim, :])  # [N,1]
        Wh2 = torch.matmul(x, self.a[self.output_dim:, :])  # [N,1]
        e = F.leaky_relu(Wh1 + Wh2.T, negative_slope=self.alpha)  # [N,N]
        return e

    def forward(self, x, adj):
        # x: [N,in_dim], adj: torch sparse COO or dense
        h = torch.matmul(x, self.weight)  # [N,out_dim]
        e = self._prepare_attentional_mechanism_input(h)  # [N,N]

        adj_dense = adj.to_dense() if adj.is_sparse else adj
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj_dense > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, training=self.training)

        h_pass = torch.matmul(attention, h)
        out = F.leaky_relu(h_pass, negative_slope=self.alpha)
        out = F.normalize(out, p=2, dim=1)

        if self.bias is not None:
            out = out + self.bias
        return out


# =========================================================
# Prompt modules
# =========================================================
class PromptBuilder(nn.Module):
    """Build heterogeneous prompts (placeholders + optional FSG).

    family/promoter prompts are implemented as trainable embeddings by default.
    You can override them by passing actual prompt tensors to PromptGRN.encode().
    """
    def __init__(self, num_genes, d_prompt, alpha=0.2):
        super().__init__()
        self.family_emb = nn.Embedding(num_genes, d_prompt)
        self.promoter_emb = nn.Embedding(num_genes, d_prompt)
        self.sim_gat = AttentionLayer(d_prompt, d_prompt, alpha=alpha)

        nn.init.xavier_uniform_(self.family_emb.weight, gain=1.414)
        nn.init.xavier_uniform_(self.promoter_emb.weight, gain=1.414)
        self.sim_gat.reset_parameters()

    def forward(self, gene_ids, adj_func=None, func_seed_feat=None, family_prompt=None, promoter_prompt=None):
        # family prompt
        p_family = self.family_emb(gene_ids) if family_prompt is None else family_prompt
        # promoter prompt
        p_promoter = self.promoter_emb(gene_ids) if promoter_prompt is None else promoter_prompt

        # similarity prompt from functional graph (FSG)
        if adj_func is None or func_seed_feat is None:
            p_sim = torch.zeros_like(p_family)
        else:
            p_sim = self.sim_gat(func_seed_feat, adj_func)

        return p_family, p_promoter, p_sim


class MediatorAggregator(nn.Module):
    """Mediator-based prompt aggregation using self-attention over 4 tokens.

    Tokens = [x0, p_family, p_promoter, p_sim], output = x0 token after attention.
    """
    def __init__(self, d_prompt, nhead=4, dropout=0.1):
        super().__init__()
        self.x0 = nn.Parameter(torch.zeros(1, 1, d_prompt))
        nn.init.normal_(self.x0, mean=0.0, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_prompt,
            nhead=nhead,
            dim_feedforward=4 * d_prompt,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)

    def forward(self, p_family, p_promoter, p_sim):
        N, d = p_family.shape
        x0 = self.x0.expand(N, 1, d)  # [N,1,d]
        tokens = torch.stack([p_family, p_promoter, p_sim], dim=1)  # [N,3,d]
        X = torch.cat([x0, tokens], dim=1)  # [N,4,d]
        H = self.encoder(X)  # [N,4,d]
        return H[:, 0, :]  # [N,d]


# =========================================================
# Encoders
# =========================================================
class LocalGNNEncoder(nn.Module):
    """Two-layer multi-head GAT-style encoder on the initial regulatory graph."""
    def __init__(self, in_dim, h1, h2, head1, head2, alpha=0.2, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        self.gat1 = nn.ModuleList([AttentionLayer(in_dim, h1, alpha) for _ in range(head1)])

        if reduction == "mean":
            h1_out = h1
        elif reduction == "concate":
            h1_out = head1 * h1
        else:
            raise ValueError("reduction must be 'mean' or 'concate'")

        self.gat2 = nn.ModuleList([AttentionLayer(h1_out, h2, alpha) for _ in range(head2)])

    def forward(self, x, adj):
        # layer 1
        if self.reduction == "concate":
            x = torch.cat([att(x, adj) for att in self.gat1], dim=1)
        else:
            x = torch.mean(torch.stack([att(x, adj) for att in self.gat1], dim=0), dim=0)
        x = F.elu(x)

        # layer 2 (mean aggregation across heads)
        x = torch.mean(torch.stack([att(x, adj) for att in self.gat2], dim=0), dim=0)
        return x


class GlobalTransformerEncoder(nn.Module):
    """Transformer encoder over all gene nodes (treat nodes as a sequence)."""
    def __init__(self, in_dim, d_model, nhead=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(self, z):
        # z: [N,in_dim] -> [1,N,d_model]
        z = self.proj(z)
        H = self.encoder(z.unsqueeze(0))
        return H.squeeze(0)  # [N,d_model]


# =========================================================
# DGI (Deep Graph Infomax) loss
# =========================================================
class DGILoss(nn.Module):
    """DGI objective using feature permutation corruption."""
    def __init__(self, d_model):
        super().__init__()
        self.M = nn.Parameter(torch.Tensor(d_model, d_model))
        nn.init.xavier_uniform_(self.M, gain=1.414)

    @staticmethod
    def corruption(x):
        idx = torch.randperm(x.size(0), device=x.device)
        return x[idx]

    @staticmethod
    def summary(h):
        return torch.sigmoid(h.mean(dim=0))

    def forward(self, h_pos, h_neg):
        s = self.summary(h_pos).unsqueeze(1)  # [d,1]
        pos_score = torch.sigmoid((h_pos @ self.M) @ s).squeeze(1)  # [N]
        neg_score = torch.sigmoid((h_neg @ self.M) @ s).squeeze(1)  # [N]

        eps = 1e-15
        loss = -0.5 * (torch.log(pos_score + eps).mean() + torch.log(1 - neg_score + eps).mean())
        return loss


# =========================================================
# Link predictor
# =========================================================
class BilinearLinkPredictor(nn.Module):
    """Bilinear TF->Target scoring: sigma(h_i^T W h_j)."""
    def __init__(self, d_model):
        super().__init__()
        self.W = nn.Parameter(torch.Tensor(d_model, d_model))
        nn.init.xavier_uniform_(self.W, gain=1.414)

    def forward(self, h, pairs):
        hi = h[pairs[:, 0]]
        hj = h[pairs[:, 1]]
        logits = torch.sum((hi @ self.W) * hj, dim=1)  # [B]
        return logits


# =========================================================
# PromptGRN: full model
# =========================================================
class PromptGRN(nn.Module):
    """PromptGRN main model.

    Pipeline:
      - local GNN on initial regulatory graph
      - build/fuse prompts via mediator
      - concat [h_local || p_final] then global Transformer
      - DGI contrastive learning on node embeddings
      - bilinear link prediction for TF->Target edges
    """
    def __init__(
        self,
        num_genes,
        x_expr_dim,
        d_prompt=128,
        gnn_h1=64,
        gnn_h2=64,
        gnn_head1=3,
        gnn_head2=3,
        alpha=0.2,
        reduction="concate",
        trans_d_model=128,
        trans_nhead=8,
        trans_layers=2,
        dropout=0.1,
    ):
        super().__init__()

        # Used to create a seed feature for similarity prompt (p_sim)
        self.expr_to_prompt = nn.Linear(x_expr_dim, d_prompt)

        # Prompt modules
        self.prompt_builder = PromptBuilder(num_genes=num_genes, d_prompt=d_prompt, alpha=alpha)
        self.mediator = MediatorAggregator(d_prompt=d_prompt, nhead=4, dropout=dropout)

        # Encoders
        self.local_gnn = LocalGNNEncoder(
            in_dim=x_expr_dim, h1=gnn_h1, h2=gnn_h2,
            head1=gnn_head1, head2=gnn_head2,
            alpha=alpha, reduction=reduction
        )
        self.global_encoder = GlobalTransformerEncoder(
            in_dim=gnn_h2 + d_prompt, d_model=trans_d_model,
            nhead=trans_nhead, num_layers=trans_layers, dropout=dropout
        )

        # Objectives/decoders
        self.dgi = DGILoss(trans_d_model)
        self.link_pred = BilinearLinkPredictor(trans_d_model)

    def encode(self, x_expr, adj_reg, gene_ids, adj_func=None, family_prompt=None, promoter_prompt=None):
        # (1) local encoding on regulatory graph
        h_local = self.local_gnn(x_expr, adj_reg)

        # (2) build prompts
        func_seed = self.expr_to_prompt(x_expr)
        p_family, p_promoter, p_sim = self.prompt_builder(
            gene_ids=gene_ids,
            adj_func=adj_func,
            func_seed_feat=func_seed,
            family_prompt=family_prompt,
            promoter_prompt=promoter_prompt
        )
        p_final = self.mediator(p_family, p_promoter, p_sim)

        # (3) global encoding with Transformer
        z = torch.cat([h_local, p_final], dim=1)
        h_final = self.global_encoder(z)
        return h_final

    def forward(self, x_expr, adj_reg, pairs, gene_ids, adj_func=None, family_prompt=None, promoter_prompt=None):
        h_final = self.encode(
            x_expr=x_expr, adj_reg=adj_reg, gene_ids=gene_ids,
            adj_func=adj_func, family_prompt=family_prompt, promoter_prompt=promoter_prompt
        )
        logits = self.link_pred(h_final, pairs)
        return logits, h_final

    def compute_losses(self, x_expr, adj_reg, pairs, labels, gene_ids, adj_func=None, lambda_dgi=0.5):
        # Supervised link prediction loss (BCE with logits)
        logits, h_pos = self.forward(x_expr, adj_reg, pairs, gene_ids, adj_func=adj_func)
        loss_grn = F.binary_cross_entropy_with_logits(logits, labels.float())

        # DGI contrastive loss with corruption
        x_corrupt = self.dgi.corruption(x_expr)
        h_neg = self.encode(x_corrupt, adj_reg, gene_ids, adj_func=adj_func)
        loss_dgi = self.dgi(h_pos, h_neg)

        loss = loss_grn + lambda_dgi * loss_dgi
        return loss, loss_grn, loss_dgi
