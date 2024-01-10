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

if __name__ == '__main__':

    lines = open('data/HumanProteome/human_proteome.txt').readlines()
    seqs = {}
    for i in range(0, len(lines), 3):
        seqid = lines[i][1:].strip()
        seq = lines[i + 1].strip()
        label = lines[i + 2].strip()
        assert len(seq) == len(label), f"{len(seq)} != {len(label)}"
        seqs[seqid] = (seq, label)

    print("Encoding sequences...")
    Path('data/HumanProteome/encoded').mkdir(exist_ok=True, parents=True)
    import time

    # encode the sequences
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

    # for name in model_names:
    #     if name in ['Rostlab/prot_t5_xl_bfd', 'Rostlab/prot_t5_xl_half_uniref50-enc']:
    #         encoder = T5Encoder(name, torch.device('cuda'))
    #     else:
    #         encoder = ProtEncoder(name, torch.device('cuda'))

    #     only_name = name.split('/')[-1]
    #     Path(f'data/HumanProteome/encoded/{only_name}').mkdir(exist_ok=True, parents=True)
    #     probs, labels, times = [], [], []
    #     for seqid, (seq, label) in tqdm(seqs.items()):
    #         p = f'data/HumanProteome/encoded/{only_name}/{seqid}.npz'
    #         if not Path(p).exists():
    #             start = time.time()
    #             emb = encoder([seq])
    #             escaped = time.time() - start
    #             # save the embedding
    #             np.savez_compressed(
    #                 p,
    #                 data=emb.cpu().numpy(),
    #                 escaped=escaped)

    args = Config('configs/ensemble_idp.ss.yaml')
    args.add_dim = 8
    args.shortcut = True
    seed_everything(args.seed)

    idp_plm = EnsemblePredictor(args).to('cuda').eval()
    if args.shortcut:
        idp_plm._load_auxiliary(None, Config('configs/ensemble_ss.yaml'))

    Path('data/HumanProteome/predictions').mkdir(exist_ok=True, parents=True)
    with torch.no_grad() and torch.cuda.amp.autocast():
        # sort the sequences by length
        seqs = {k: v for k, v in sorted(seqs.items(), key=lambda item: len(item[1][0]))}
        for seqid, (seq, label) in tqdm(seqs.items()):
            embs = {}
            times = {}
            p = f'data/HumanProteome/predictions/{seqid}.npz'
            if Path(p).exists():
                continue
            for name in model_names:
                only_name = name.split('/')[-1]
                z = np.load(
                    f'/home/ol/hdd2/idp-plm/data/HumanProteome/encoded/{only_name}/{seqid}.npz', allow_pickle=True)
                embs[only_name] = torch.from_numpy(z['data']).to('cuda')
                times[only_name] = z['escaped']

            start = time.time()
            logits = idp_plm(embs)
            inf_time = time.time() - start

            prob = logits.mean(-1).softmax(-1)[..., 1].detach().cpu().numpy()

            np.savez_compressed(
                p,
                prob=prob,
                label=label,
                seq=seq,
                time=sum(times.values()) + inf_time,
            )
