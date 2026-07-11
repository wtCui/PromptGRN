import os
import argparse
import numpy as np
import pandas as pd
from utils import Network_Statistic

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ratio', type=float, default=0.7, help='Train ratio for positive edges.')
    p.add_argument('--num', type=int, default=500, help='Network scale.')
    p.add_argument('--p_val', type=float, default=0.5, help='Probability for single-edge TF to go to train.')
    p.add_argument('--data', type=str, default='hESC', help='Data type.')
    p.add_argument('--net', type=str, default='Specific', help='Network type.')
    p.add_argument('--out_dir', type=str, default='./Demo/Train_validation_test', help='Output base directory.')
    return p.parse_args()

def build_pos_dict(label_df):
    pos = {}
    for tf, tg in label_df[['TF', 'Target']].values:
        pos.setdefault(int(tf), []).append(int(tg))
    for k in pos:
        pos[k] = list(sorted(set(pos[k])))
    return pos

def sample_neg_for_tf(tf, gene_set, forbidden, k):
    neg = []
    while len(neg) < k:
        g = int(np.random.choice(gene_set))
        if g == tf:
            continue
        if (tf, g) in forbidden:
            continue
        neg.append(g)
    return neg

def split_promptgrn(label_file, gene_file, tf_file, out_dir, density, ratio=0.7, p_val=0.5):
    gene_set = pd.read_csv(gene_file, index_col=0)['index'].values.astype(np.int64)
    tf_set = pd.read_csv(tf_file, index_col=0)['index'].values.astype(np.int64)
    label_df = pd.read_csv(label_file, index_col=0)[['TF','Target']]

    pos_dict = build_pos_dict(label_df)

    train_pos, val_pos, test_pos = {}, {}, {}
    for tf, targets in pos_dict.items():
        n = len(targets)
        if n == 1:
            if np.random.rand() < p_val:
                train_pos[tf] = targets
            else:
                test_pos[tf] = targets
        elif n == 2:
            np.random.shuffle(targets)
            train_pos[tf] = [targets[0]]
            test_pos[tf] = [targets[1]]
        else:
            targets = targets.copy()
            np.random.shuffle(targets)
            n_train = int(n * ratio)
            n_val = max(1, int(n * 0.1))
            train_pos[tf] = targets[:n_train]
            val_pos[tf] = targets[n_train:n_train+n_val]
            test_pos[tf] = targets[n_train+n_val:]

    all_pos = set((tf, tg) for tf, tgs in pos_dict.items() for tg in tgs)

    # Train set (1:1 negatives per TF)
    train_pairs, train_labels = [], []
    for tf, tgs in train_pos.items():
        for tg in tgs:
            train_pairs.append([tf, tg]); train_labels.append(1)
        negs = sample_neg_for_tf(tf, gene_set, all_pos, len(tgs))
        for tg in negs:
            train_pairs.append([tf, tg]); train_labels.append(0)

    # Val set (1:1 negatives per TF)
    val_pairs, val_labels = [], []
    for tf, tgs in val_pos.items():
        for tg in tgs:
            val_pairs.append([tf, tg]); val_labels.append(1)
        negs = sample_neg_for_tf(tf, gene_set, all_pos, len(tgs))
        for tg in negs:
            val_pairs.append([tf, tg]); val_labels.append(0)

    # Test set (density-controlled global negatives)
    test_pairs, test_labels = [], []
    for tf, tgs in test_pos.items():
        for tg in tgs:
            test_pairs.append([tf, tg]); test_labels.append(1)

    num_pos = len(test_labels)
    num_neg = int(num_pos / density - num_pos) if density is not None else num_pos

    forbidden = set(tuple(x) for x in train_pairs + val_pairs + test_pairs)
    while (len(test_labels) < num_pos + num_neg):
        tf = int(np.random.choice(tf_set))
        tg = int(np.random.choice(gene_set))
        if tf == tg:
            continue
        if (tf, tg) in forbidden:
            continue
        test_pairs.append([tf, tg]); test_labels.append(0)
        forbidden.add((tf, tg))

    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame({"TF":[p[0] for p in train_pairs], "Target":[p[1] for p in train_pairs], "Label":train_labels}).to_csv(
        os.path.join(out_dir, "Train_set.csv"), index=False
    )
    pd.DataFrame({"TF":[p[0] for p in val_pairs], "Target":[p[1] for p in val_pairs], "Label":val_labels}).to_csv(
        os.path.join(out_dir, "Validation_set.csv"), index=False
    )
    pd.DataFrame({"TF":[p[0] for p in test_pairs], "Target":[p[1] for p in test_pairs], "Label":test_labels}).to_csv(
        os.path.join(out_dir, "Test_set.csv"), index=False
    )

def main():
    args = parse_args()
    density = Network_Statistic(data_type=args.data, net_scale=args.num, net_type=args.net)

    base = os.getcwd()
    tf_file = f"{base}/{args.net} Dataset/{args.data}/TFs+{args.num}/TF.csv"
    gene_file = f"{base}/{args.net} Dataset/{args.data}/TFs+{args.num}/Target.csv"
    label_file = f"{base}/{args.net} Dataset/{args.data}/TFs+{args.num}/Label.csv"

    out_dir = os.path.join(args.out_dir, args.net, f"{args.data}_{args.num}")
    os.makedirs(out_dir, exist_ok=True)

    split_promptgrn(label_file, gene_file, tf_file, out_dir, density, ratio=args.ratio, p_val=args.p_val)
    print(f"Saved Train/Validation/Test CSVs to: {out_dir}")

if __name__ == "__main__":
    main()
