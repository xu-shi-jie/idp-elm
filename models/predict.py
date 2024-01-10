from pathlib import Path

import numpy as np
import pytorch_lightning as L
import streamlit as st
import torch
import torch.nn as nn
from dataloader import CaidDataModule, flDPnnDataModule
from encoder import ProtEncoder, T5Encoder
from ensemble import EnsemblePredictor
from nets import Head
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from tqdm import tqdm
from utils import Config, EsmModelInfo, seed_everything

args = Config('configs/ensemble_idp.yaml')
args.add_dim = 0
seed_everything(args.seed)

ver = 'v23.4.24'

idp_plm = EnsemblePredictor(args).to('cuda:1').eval()
model_names = [
    'facebook/esm1b_t33_650M_UR50S',
    'facebook/esm2_t6_8M_UR50D',
    'facebook/esm2_t12_35M_UR50D',
    'facebook/esm2_t30_150M_UR50D',
    'facebook/esm2_t33_650M_UR50D',
    'facebook/esm2_t36_3B_UR50D',
    'facebook/esm2_t48_15B_UR50D',
    'Rostlab/prot_t5_xl_bfd',
    'Rostlab/prot_t5_xl_half_uniref50-enc',
]


def get_encoders():
    encoders = {}
    for full_name in model_names:
        name = full_name.split('/')[-1]
        print("Loading encoder: ", name)
        if name == 'prot_t5_xl_bfd':
            encoder = T5Encoder(full_name, torch.device('cuda'))
        elif name == 'prot_t5_xl_half_uniref50-enc':
            encoder = T5Encoder(full_name, torch.device('cuda'))
        else:
            encoder = ProtEncoder(full_name, torch.device('cuda:1'))
        encoders[name] = encoder
    print("Done loading encoders.")
    return encoders


lines = open('data/HumanProteome/human_proteome.txt').readlines()
seqs = []
for i in range(0, len(lines), 3):
    seqid = lines[i][1:].strip()
    seq = lines[i + 1].strip()
    label = lines[i + 2].strip()
    assert len(seq) == len(label), f"{len(seq)} != {len(label)}"
    seqs.append((seqid, seq, label))

# sort by length of sequence
seqs = sorted(seqs, key=lambda x: len(x[1]))
# remove sequences longer equal than 14507
seqs = [s for s in seqs if len(s[1]) < 14507]
print("Number of sequences: ", len(seqs))


encoders = get_encoders()
Path('data/HumanProteome/predictions').mkdir(exist_ok=True, parents=True)

import time

probs = []
labels = []
times = []
for seqid, seq, label in (pbar := tqdm(seqs, total=len(seqs))):
    embs = {}
    p = f'data/HumanProteome/predictions/{seqid}.npz'
    if Path(p).exists():
        continue

    start = time.time()

    for name, model in encoders.items():
        emb = model([seq])
        embs[name] = emb.to('cuda:0').detach()

    with torch.no_grad() and torch.cuda.amp.autocast():
        logits = idp_plm(embs)
        prob = logits.sum(-1).softmax(-1)[..., 1].detach().cpu().numpy()

        times.append((len(seq), time.time() - start))
        assert len(prob) == len(label), f"{len(prob)} != {len(label)}"
        probs.append(prob)
        labels.append(np.array([int(a) for a in label]))

    np.savez_compressed(
        p,
        prob=prob,
        label=label,
        seq=seq,
        time=time.time() - start,
    )
