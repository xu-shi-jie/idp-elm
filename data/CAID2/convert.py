from pathlib import Path

import numpy as np


def read_actual():
    lines = open('data/CAID2_clean.txt').read().splitlines()
    # lines = open('disorder_pdb_2.txt').read().splitlines()
    # lines = open('../data/CAID.txt').read().splitlines()
    seqs = {}
    for i in range(0, len(lines), 3):
        seqid = lines[i].strip()[1:]
        seq = lines[i + 1].strip()
        dis = np.array([int(i) for i in lines[i + 2]])
        seqs[seqid] = (seq, dis)
    return seqs


def read_results(p):
    lines = [l for l in open(p).read().split('>') if l.strip()]
    seqs = {}
    for line in lines:
        line = line.splitlines()
        seqid = line[0].strip()
        aa_idx, aa, prob, pred = zip(*[l.split('\t') for l in line[1:]])
        seq = ''.join(aa)
        if ''.join(prob).strip() == '':
            prob = None
        else:
            prob = np.array([float(i) for i in prob])
        pred = np.array([int(i) for i in pred])
        seqs[seqid] = (seq, prob, pred)
    return seqs


import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (auc, f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from tqdm import tqdm

fig = plt.figure(figsize=(10, 10))

df = pd.DataFrame(columns=['Predictor', 'ROC AUC', 'F1', 'MCC', 'Recall', 'Precision', 'Failed'])


actual = read_actual()

failed_set = set()
names = ['SPOT-Disorder2', 'IDP-PLM', 'SPOT-Disorder-Single']
for file in tqdm(list(Path('data/CAID2/predictions').glob("*.caid"))):
    predicotr_name = file.stem
    if predicotr_name not in names:
        continue
    predictions = read_results(file)

    probs, preds, labels = [], [], []
    failed = 0
    for seqid in actual:
        # if seqid in failed_set:
        #     continue
        seq, dis = actual[seqid]

        if seqid not in predictions:
            # print(f"No prediction for {seqid} in {file}")
            failed_set.add(seqid)
            failed += 1
            continue
        seq2, prob, pred = predictions[seqid]
        assert len(prob) == len(pred) == len(seq), f"{file}: {seqid} {len(prob)} {len(pred)} {len(seq)}"
        assert seq == seq2, f"{file}: {seqid} {len(seq)} {len(seq2)}"

        probs.append(prob)
        preds.append(pred)
        labels.append(dis)

    probs = np.concatenate(probs)
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    mask = (labels != 2).astype(bool)
    # probs = probs[mask]
    # preds = preds[mask]
    # labels = labels[mask]
    labels[~mask] = 0

    # if predicotr_name == 'IDP-PLM':
    #     preds = probs > 0.00001

    df.loc[len(df)] = [predicotr_name, roc_auc_score(labels, probs), f1_score(labels, preds), matthews_corrcoef(
        labels, preds), recall_score(labels, preds), precision_score(labels, preds), failed]

    # if predicotr_name in chosen_list:
    #     order.append(chosen_list.index(predicotr_name))
    # add ROC curve to figure
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, label=f'{predicotr_name} ($A={roc_auc:.2f}$)', color=pal[len(df) - 1])


# add our results
# df.loc[len(df)] = ['IDP-PLM', 0.8448, 0.5504, 0.4362, 0.7265, 0.7101]

# draw legend with the name of chosen_list
handles, ls = plt.gca().get_legend_handles_labels()

# add title, x and y labels
plt.title('Receiver Operating Characteristic', fontsize=16)
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
# set x axis length same as y axis
plt.gca().set_aspect('equal', adjustable='box')
# draw the line for AUC = 0.5, 0.6, 0.7, 0.8, 0.9
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.show()

print(failed_set)

# sort df by ROC AUC togather with other metrics
df = df.sort_values(by=['ROC AUC', 'F1', 'MCC', 'Recall', 'Precision'], ascending=False)
# print the first 10 rows' predictor as a list
print(df['Predictor'].tolist())
# show with 3 decimal places
print(df.round(4))
