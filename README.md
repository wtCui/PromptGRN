# PromptGRN

PromptGRN pipeline:

1) Build an initial (weakly-supervised) regulatory graph from **positive** TF→Target edges.
2) Build functional prompts from heterogeneous priors:
   - Gene family prompt
   - Promoter sequence prompt
   - GO/Pathway similarity prompt
3) Fuse prompts with a learnable **mediator** and self-attention.
4) Encode local structure with a **GNN (GAT)** and model global dependencies with a **Transformer**.
5) Add **DGI** contrastive learning for robustness (feature permutation corruption).
6) Predict edges using a **bilinear** decoder and train with a joint loss: BCE + λ·DGI.


## Project Structure

  - `models.py`         Model definition (PromptGRN + submodules)
  - `data_utils.py`     Loading CSVs, building adjacency, dynamic negative sampling, normalization
  - `metrics.py`        AUC/AUPR evaluation helpers (compatible with your previous code style)
  - `train.py`          Training entry (dynamic negative sampling + DGI)
  - `infer.py`          Inference: score TF→gene pairs and export ranked edges
  - `split_dataset.py`  PromptGRN-style dataset splitting (positive-first split + per-split negatives)

## Data Assumptions

- Expression feature file: CSV where rows correspond to genes (N genes) and columns are features (d).
  The loader standardizes features (z-score) by default.
- TF list file: CSV with a column named `index` (integer gene indices) of TF genes.
- Target list file: CSV with column `index` (integer indices) for candidate target genes.
- Label file: CSV with columns `TF` and `Target` (positive TF→Target pairs).
- Train/Val/Test CSVs: each row `[TF, Target, Label]` with Label in {0,1}.

## Quick Start

1) Split dataset (generates Train/Validation/Test CSVs):
```bash
python scripts/split_dataset.py --data hESC --net Specific --num 500 --ratio 0.7 --p_val 0.5 --out_dir ./Demo/Train_validation_test
```

2) Train PromptGRN:
```bash
python -m promptgrn.train   --exp_file ./Demo/hESC/TFs+500/BL--ExpressionData.csv   --tf_file  ./Demo/hESC/TFs+500/TF.csv   --train_file ./Demo/Train_validation_test/Specific/hESC_500/Train_set.csv   --val_file   ./Demo/Train_validation_test/Specific/hESC_500/Validation_set.csv   --save_dir ./model   --epochs 100 --lr 3e-3 --lambda_dgi 0.5
```

3) Inference (score all TF→gene pairs and export top-K):
```bash
python -m promptgrn.infer   --exp_file ./Demo/hESC/TFs+500/BL--ExpressionData.csv   --tf_file  ./Demo/hESC/TFs+500/TF.csv   --model_ckpt ./model/PromptGRN.pt   --out_csv ./Result/predicted_edges.csv   --topk 200000
```


