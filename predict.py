import sys

sys.path.append('models')

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as L
import seaborn as sns
import torch
import torch.nn as nn
from loguru import logger
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from tqdm import tqdm

from models.dataloader import CaidDataModule, flDPnnDataModule
from models.encoder import ProtEncoder, T5Encoder
from models.ensemble import EnsemblePredictor
from models.nets import Head
from models.utils import Config, EsmModelInfo, seed_everything

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

def print_help():
    ver = '1.0'
    print('-' * 80)
    print(f'                 IDP-ELM v{ver}, Developed by Shijie Xu, 2023                 \n')
    print(f'''If you use this tool, please cite paper: 
    Xu, Shijie, and Akira Onoda. "Accurate and Fast Prediction of Intrinsically 
    Disordered Protein by Multiple Protein Language Models and Ensemble 
    Learning." Journal of Chemical Information and Modeling (2023).\n''')
    print('''Usuage example: 
    python predict.py --fasta example.fasta --output_dir predictions\n''')
    print(''' FASTA file should start with seq ID followed by one-line seq
    sequences and mupltiple sequences are allowed.''')
    print('''The output files will be saved to 
    output_dir/1.idpelm.txt, output_dir/2.idpelm.txt, ... 
    according to the order of sequences in the input fasta file.\n''')
    print('Each output file contains five lines: ')
    print('1. The sequence ID')
    print('2. The sequence')
    print('3. The probability of being disordered')
    print('4. The probability of being disordered linker')
    print('5. The probability of being disordered protein binding')
    print('-' * 80)


if __name__ == '__main__':
    idp_plm_args = Config('configs/ensemble_idp.yaml')
    linker_args = Config('configs/ensemble_linker.yaml')
    pb_args = Config('configs/ensemble_prot.yaml')

    idp_plm_args.add_dim = 0
    linker_args.add_dim = 0
    pb_args.add_dim = 2

    seed_everything(idp_plm_args.seed)
    print_help()

    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta', type=str, default='example.fasta')
    parser.add_argument('--output_dir', type=str, default='predictions')
    cmd_args = parser.parse_args()

    lines = open(cmd_args.fasta).readlines()

    seqs = []
    for i in range(0, len(lines), 2):
        seqid = lines[i][1:].strip()
        seq = lines[i + 1].strip()
        seqs.append((seqid, seq))

    # sort by length of sequence
    seqs = sorted(seqs, key=lambda x: len(x[1]))

    print(f"Number of sequences: {len(seqs)}\nThe longest sequence has {len(seqs[-1][1])} residues.\n")

    encoders = get_encoders()
    output_dir = Path(cmd_args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"Results will be saved to {output_dir}")

    p = output_dir / (Path(cmd_args.fasta).stem + '.idpelm.txt')
    seqid_embs = []
    for i, (seqid, seq) in (pbar := tqdm(enumerate(seqs), total=len(seqs))):
        embs = {}
        for name, model in encoders.items():
            logger.info(f'Performing PLM {name} on {seqid} by {model.model.device}')
            emb = model([seq])
            embs[name] = emb.detach().to('cuda')
        seqid_embs.append((seqid, seq, embs))

    # delete encoders
    del encoders
    torch.cuda.empty_cache()
    # load new  models
    idp_plm = EnsemblePredictor(idp_plm_args).to('cuda').eval()
    linker = EnsemblePredictor(linker_args).to('cuda').eval()
    linker._load_auxiliary(Config('configs/ensemble_idp.yaml'), Config('configs/ensemble_ss.yaml'))
    pb = EnsemblePredictor(pb_args).to('cuda').eval()
    pb._load_auxiliary(Config('configs/ensemble_idp.yaml'), Config('configs/ensemble_ss.yaml'))
    results = []
    for seqid, seq, embs in seqid_embs:
        with torch.no_grad() and torch.cuda.amp.autocast():
            idp_logits = idp_plm(embs)
            idp_prob = idp_logits.sum(-1).softmax(-1)[..., 1].detach().cpu().numpy()
            linker_logits = linker(embs)
            linker_prob = linker_logits.sum(-1).softmax(-1)[..., 1].detach().cpu().numpy()
            pb_logits = pb(embs)
            pb_prob = pb_logits.sum(-1).softmax(-1)[..., 1].detach().cpu().numpy()

            results.append((seqid, seq, idp_prob, linker_prob, pb_prob))

    with open(p, 'a') as f:
        for seqid, seq, idp_prob, linker_prob, pb_prob in results:
            f.write(f'>{seqid}\n{seq}\n')
            f.write('\t'.join([f'{p:.4f}' for p in idp_prob]) + '\n')
            f.write('\t'.join([f'{p:.4f}' for p in linker_prob]) + '\n')
            f.write('\t'.join([f'{p:.4f}' for p in pb_prob]) + '\n')