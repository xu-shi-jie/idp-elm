
import time
from pathlib import Path

import numpy as np
import pytorch_lightning as L
import torch
import torch.nn as nn
from dataloader import (ApodDataModule, Caid2DataModule, CaidDataModule,
                        DeepDisoBindDataModule, DisoLipPredDataModule,
                        DisProt2022DecDataModule, NetSurfP30DataModule,
                        VariousIdpTestDataModule, flDPnnDataModule)
from nets import Head, LinkerPredictor
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)
from tqdm import tqdm
from utils import Config, EsmModelInfo, seed_everything

import wandb


class EnsemblePredictor(nn.Module):
    def __init__(self, args: Config):
        super().__init__()
        self.args = args

        self.predictors = {}
        self.plms = [
            'esm1b_t33_650M_UR50S',
            'esm2_t6_8M_UR50D',
            'esm2_t12_35M_UR50D',
            'esm2_t30_150M_UR50D',
            'esm2_t33_650M_UR50D',
            'esm2_t36_3B_UR50D',
            'esm2_t48_15B_UR50D',
            'prot_t5_xl_bfd',
            'prot_t5_xl_half_uniref50-enc',
        ]
        for model_name in self.plms:
            model_info = EsmModelInfo(model_name)
            head = Head(
                type=args.__getattr__(model_name)['head_type'],
                input_size=model_info['dim'] + args.add_dim,
                hidden_size=args.__getattr__(model_name)['hidden_size'],
                num_layers=args.__getattr__(model_name)['num_layers'],
                dropout=args.__getattr__(model_name)['dropout'],
                output_size=8 if args.task == 'ss' else 2,
            )
            ckpt_path = f'weights/{args.task}{".ss" if args.shortcut else ""}/' + model_name + '.ckpt'
            state_dict = torch.load(ckpt_path)['state_dict']
            # replace 'head.' with ''
            state_dict = {k.replace('head.', ''): v for k, v in state_dict.items() if k.startswith('head')}
            head.load_state_dict(state_dict)
            head.to(args.device)
            head.eval()
            self.predictors[model_name] = head

    def _load_auxiliary(self, idp_config: str, ss_config: str):
        self.ss_head = {}
        self.idp_head = {}
        for model_name in self.plms:
            if self.args.task != 'idp':
                idp_head = Head(
                    type=idp_config.__getattr__(model_name)['head_type'],
                    input_size=EsmModelInfo(model_name)['dim'],
                    hidden_size=idp_config.__getattr__(model_name)['hidden_size'],
                    num_layers=idp_config.__getattr__(model_name)['num_layers'],
                    dropout=idp_config.__getattr__(model_name)['dropout'],
                    output_size=2,
                )
                p = f'weights/idp/' + model_name + '.ckpt'
                state_dict = torch.load(p)['state_dict']
                # replace 'head.' with ''
                state_dict = {k.replace('head.', ''): v for k, v in state_dict.items()}
                idp_head.load_state_dict(state_dict)
                idp_head.to(self.args.device)
                idp_head.eval()
                self.idp_head[model_name] = idp_head

            ss_head = Head(
                type=ss_config.__getattr__(model_name)['head_type'],
                input_size=EsmModelInfo(model_name)['dim'],
                hidden_size=ss_config.__getattr__(model_name)['hidden_size'],
                num_layers=ss_config.__getattr__(model_name)['num_layers'],
                dropout=ss_config.__getattr__(model_name)['dropout'],
                output_size=8,
            )
            p = f'weights/ss/' + model_name + '.ckpt'
            state_dict = torch.load(p)['state_dict']
            # replace 'head.' with ''
            state_dict = {k.replace('head.', ''): v for k, v in state_dict.items()}
            ss_head.load_state_dict(state_dict)
            ss_head.to(self.args.device)
            ss_head.eval()
            # do not set ss_head and idp_head to a submodel of self, because it will be saved in the checkpoint
            self.ss_head[model_name] = ss_head

    def forward(self, x):
        preds = []
        for model_name in x:
            if self.args.add_dim == 8:
                ss = self.ss_head[model_name](x[model_name]).detach()
                pred = self.predictors[model_name](torch.cat([x[model_name], ss], dim=-1))
            elif self.args.add_dim == 2:
                idp = self.idp_head[model_name](x[model_name]).detach()
                pred = self.predictors[model_name](torch.cat([x[model_name], idp], dim=-1))
            elif self.args.add_dim == 10:
                idp = self.idp_head[model_name](x[model_name]).detach()
                ss = self.ss_head[model_name](x[model_name]).detach()
                pred = self.predictors[model_name](torch.cat([x[model_name], idp, ss], dim=-1))
            else:
                pred = self.predictors[model_name](x[model_name])
            preds.append(pred)
        return torch.stack(preds, dim=-1)


def evaluate_idp(emb_path, test, shortcut=False):
    if shortcut:
        args = Config('configs/ensemble_idp.ss.yaml')
        args.add_dim = 8
    else:
        args = Config('configs/ensemble_idp.yaml')
        args.add_dim = 0
    args.task = 'idp'
    args.shortcut = shortcut
    args.test = test

    seed_everything(args.seed)
    write_out = False
    if test == 'fldpnn':
        test_ds = flDPnnDataModule(batch_size=args.batch_size, num_workers=args.num_workers, seed=args.seed)
    elif test in ['casp', 'disorder723']:
        test_ds = VariousIdpTestDataModule(batch_size=args.batch_size, num_workers=args.num_workers)
    elif test == 'disprot':
        test_ds = DisProt2022DecDataModule(batch_size=args.batch_size, num_workers=args.num_workers)
    elif test == 'caid':
        test_ds = CaidDataModule(batch_size=args.batch_size, num_workers=args.num_workers)
        f = open('others/CAID_predictions/D034_IDP-PLM.out', 'w')
        write_out = True
    elif test == 'caid2':
        test_ds = Caid2DataModule(batch_size=args.batch_size, num_workers=args.num_workers)
        f = open('data/CAID2/predictions/IDP-PLM.caid', 'w')
        write_out = True
    else:
        raise NotImplementedError
    model = EnsemblePredictor(args).to(args.device)
    if shortcut:
        model._load_auxiliary(None, Config('configs/ensemble_ss.yaml'))

    for d in [test_ds]:
        predictions, labels = [], []
        fully_dis_pred, fully_dis_label = [], []
        for batch in tqdm(d.test_dataloader()
                          if test in ['fldpnn', 'disprot', 'caid', 'caid2'] else d.test_dataloader(test)):
            ids, seqs, dis = zip(*batch)
            for i, id in enumerate(ids):
                assert len(seqs[i]) == len(dis[i]), f'{id} {len(seqs[i])} {len(dis[i])}'
                embs = {}
                for p in emb_path:
                    model_name = p.split('/')[-1]
                    embs[model_name] = torch.load(p + '/' + id[1:] + '.pt').to(args.device)
                with torch.no_grad():
                    logits = model(embs)
                disorder = torch.tensor([int(d) for d in dis[i]]).to(args.device)

                logits = logits.mean(-1)[:len(disorder)]
                prob = logits.softmax(-1)[..., 1]

                assert len(logits) == len(disorder), f'{len(logits)} != {len(disorder)}'

                if write_out:
                    # write to .out format
                    f.write(id + '\n')
                    for aa_idx, l in enumerate(logits):
                        f.write(f'{aa_idx+1}\t{seqs[i][aa_idx]}\t{prob[aa_idx].item()}\t{l.argmax().item()}\n')

                predictions.append(prob.detach())
                labels.append(disorder.detach())

                # check if 95% of the sequence is disorder, if more or equal than 5 predictors agree, then it is fully disordered
                # fully_dis_label.append(disorder.sum() / len(disorder) > 0.95)
                # _scores = (logits.argmax(-2).sum(0) / logits.shape[0] > 0.95)
                # fully_dis_pred.append(_scores.sum() > (0 if len(emb_path) == 1 else 5))

        predictions = torch.cat(predictions)
        labels = torch.cat(labels)
        assert len(predictions) == len(labels), f'{len(predictions)} != {len(labels)}'
        mask = (labels == 0) | (labels == 1)
        predictions = predictions[mask].cpu().numpy()
        labels = labels[mask].cpu().numpy()

        auc = roc_auc_score(labels, predictions)
        f1 = f1_score(labels, predictions > 0.5)
        mcc = matthews_corrcoef(labels, predictions > 0.5)
        bacc = balanced_accuracy_score(labels, predictions > 0.5)
        sn = recall_score(labels, predictions > 0.5)
        sp = recall_score(labels, predictions > 0.5, pos_label=0)
        recall = recall_score(labels, predictions > 0.5)
        precision = precision_score(labels, predictions > 0.5)

        print("AUC: {:.4f}, MCC: {:.4f}, BACC: {:.4f}, SN: {:.4f}, SP: {:.4f}, F1: {:.4f}, Recall: {:.4f}, Precision: {:.4f}".format(
            auc, mcc, bacc, sn, sp, f1, recall, precision))

        # fully_dis_label = torch.tensor(fully_dis_label).cpu().numpy()
        # fully_dis_pred = torch.tensor(fully_dis_pred).cpu().numpy()
        # print("F1: {:.4f}, MCC: {:.4f}".format(
        #     f1_score(fully_dis_label, fully_dis_pred),
        #     matthews_corrcoef(fully_dis_label, fully_dis_pred),
        # ))
        # print(fully_dis_label.sum(), len(fully_dis_label))


def evaluate_linker(emb_path, shortcut=False):
    if shortcut:
        args = Config('configs/ensemble_linker.ss.yaml')
        args.add_dim = 8
    else:
        args = Config('configs/ensemble_linker.yaml')
        args.add_dim = 0
    args.task = 'linker'
    seed_everything(args.seed)

    dm = ApodDataModule(batch_size=args.batch_size, num_workers=args.num_workers)
    model = EnsemblePredictor(args).to(args.device)
    model._load_auxiliary(Config('configs/ensemble_idp.yaml'), Config('configs/ensemble_ss.yaml'))

    tokens = {'1': 1, '0': 0, 'x': 2}

    for t in ['te82']:  # , 'te64']:
        predictions, labels = [], []
        for batch in tqdm(dm.test_dataloader(t)):
            ids, seqs, label = zip(*batch)
            for i, id in enumerate(ids):
                embs = {}
                for p in emb_path:
                    model_name = p.split('/')[-1]
                    embs[model_name] = torch.load(p + '/' + id[1:] + '.pt').to(args.device)
                logits = model(embs)

                linker = torch.tensor([tokens[d] for d in label[i]]).to(args.device)
                mask = linker != 2

                predictions.append(logits.sum(-1)[mask].detach())
                labels.append(linker[mask].detach())

        predictions = torch.cat(predictions)
        predictions = predictions.softmax(-1)[..., 1].cpu().numpy()

        labels = torch.cat(labels).cpu().numpy()
        auc = roc_auc_score(labels, predictions)
        thr = 0.16
        f1 = f1_score(labels, predictions > thr)
        mcc = matthews_corrcoef(labels, predictions > thr)
        rec = recall_score(labels, predictions > thr)
        pre = precision_score(labels, predictions > thr)
        print("[{}] AUC: {:.4f}, F1: {:.4f}, MCC: {:.4f}, REC: {:.4f}, PRE: {:.4f}".format(t, auc, f1, mcc, rec, pre))


def evaluate_rdp(task: str, emb_path: str, shortcut=False):
    if shortcut:
        args = Config(f'configs/ensemble_{task}.ss.yaml')
        args.add_dim = 10
    else:
        args = Config(f'configs/ensemble_{task}.yaml')
        args.add_dim = 2
    args.task = task
    seed_everything(args.seed)

    dm = DeepDisoBindDataModule(batch_size=args.batch_size, num_workers=args.num_workers)
    model = EnsemblePredictor(args).to(args.device)
    model._load_auxiliary(Config('configs/ensemble_idp.yaml'), Config('configs/ensemble_ss.yaml'))

    tokens = {'0': 0, '1': 1}

    predictions, labels = [], []
    for batch in tqdm(dm.test_dataloader()):
        ids, seqs, dis, prot, dna, rna = zip(*batch)
        label = prot if task == 'prot' else dna if task == 'dna' else rna
        for i, id in enumerate(ids):
            embs = {}
            for p in emb_path:
                model_name = p.split('/')[-1]
                embs[model_name] = torch.load(p + '/' + id[1:] + '.pt').to(args.device)
            logits = model(embs)

            rdp = torch.tensor([tokens[d] for d in label[i]]).to(args.device)

            predictions.append(logits.sum(-1).detach())
            labels.append(rdp.detach())

    predictions = torch.cat(predictions)
    predictions = predictions.softmax(-1)[..., 1].cpu().numpy()

    labels = torch.cat(labels).cpu().numpy()
    auc = roc_auc_score(labels, predictions)

    threshold = np.linspace(0, 1, 1000)
    specificity = []
    for t in threshold:
        specificity.append(((predictions[labels == 0] < t).sum() / (labels == 0).sum()).item())
    # set threshold to the point where specificity is 0.8
    t = threshold[np.argmin(np.abs(np.array(specificity) - 0.8))]

    f1 = f1_score(labels, predictions > t)
    mcc = matthews_corrcoef(labels, predictions > t)
    rec = recall_score(labels, predictions > t)
    pre = precision_score(labels, predictions > t)
    sn = recall_score(labels, predictions > t, pos_label=1)
    sp = recall_score(labels, predictions > t, pos_label=0)
    print(
        "AUC: {:.4f}, F1: {:.4f}, MCC: {:.4f}, REC: {:.4f}, PRE: {:.4f}, SN: {:.4f}, SP: {:.4f}".format(
            auc, f1, mcc, rec, pre, sn, sp))


def evaluate_ss(emb_path):
    args = Config(f'configs/ensemble_ss.yaml')
    args.add_dim = 0
    seed_everything(args.seed)

    dm = NetSurfP30DataModule(batch_size=args.batch_size, num_workers=args.num_workers)
    model = EnsemblePredictor(args).to(args.device)

    tokens = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8}

    for t in [
        dm.test_dataloader('casp12'),
        dm.test_dataloader('cb513'),
        dm.test_dataloader('ts115')
    ]:
        predictions, labels = [], []
        for batch in tqdm(t):
            ids, seqs, label = zip(*batch)
            for i, id in enumerate(ids):
                embs = {}
                for p in emb_path:
                    model_name = p.split('/')[-1]
                    embs[model_name] = torch.load(p + '/' + id[1:] + '.pt').to(args.device)
                logits = model(embs)

                ss = torch.tensor([tokens[d] for d in label[i]]).to(args.device)
                mask = (ss != 8).bool()

                predictions.append(logits.sum(-1)[:len(ss)][mask].detach())
                labels.append(ss[mask].detach())

        predictions = torch.cat(predictions)
        predictions = predictions.argmax(-1).cpu().numpy()

        labels = torch.cat(labels).cpu().numpy()

        acc = accuracy_score(labels, predictions)
        print("ACC: {:.4f}".format(acc))

        def q8_to_q3(a):  # Q8 GHIBESTC (Q8 -> Q3: HHHEECCC)
            return np.array([0 if x in [0, 1, 2] else 1 if x in [3, 4] else 2 for x in a])
        acc3 = accuracy_score(q8_to_q3(labels), q8_to_q3(predictions))
        print("ACC3: {:.4f}".format(acc3))


if __name__ == '__main__':
    def str2bool(s: str):
        return s.lower() in ['true', '1', 't', 'y', 'yes']

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='idp')
    parser.add_argument('--test', type=str, default='fldpnn')
    parser.add_argument('--shortcut', type=str2bool, default=True)
    args = parser.parse_args()
    emb_path = [
        f'embs/{"rdp" if args.task=="prot" else args.task}/facebook/esm1b_t33_650M_UR50S',
        f'embs/{"rdp" if args.task=="prot" else args.task}/facebook/esm2_t6_8M_UR50D',
        f'embs/{"rdp" if args.task=="prot" else args.task}/facebook/esm2_t12_35M_UR50D',
        f'embs/{"rdp" if args.task=="prot" else args.task}/facebook/esm2_t30_150M_UR50D',
        f'embs/{"rdp" if args.task=="prot" else args.task}/facebook/esm2_t33_650M_UR50D',
        f'embs/{"rdp" if args.task=="prot" else args.task}/facebook/esm2_t36_3B_UR50D',
        f'embs/{"rdp" if args.task=="prot" else args.task}/facebook/esm2_t48_15B_UR50D',
        f'embs/{"rdp" if args.task=="prot" else args.task}/Rostlab/prot_t5_xl_bfd',
        f'embs/{"rdp" if args.task=="prot" else args.task}/Rostlab/prot_t5_xl_half_uniref50-enc',
    ]
    if args.task == 'idp':
        print(args)
        for p in emb_path:
            print(p)
            evaluate_idp([p], args.test, shortcut=args.shortcut)
        print("Ensemble")
        start = time.time()
        evaluate_idp(emb_path, args.test, shortcut=args.shortcut)
        print(f"Ensemble time: {time.time() - start} sec")
    elif args.task == 'ss':
        for p in emb_path:
            print(p)
            evaluate_ss([p])
    elif args.task == 'linker':
        for p in emb_path:
            print(p)
            evaluate_linker([p], shortcut=args.shortcut)
        print("Ensemble")
        evaluate_linker(emb_path, shortcut=args.shortcut)
    elif args.task == 'prot':
        for p in emb_path:
            print(p)
            evaluate_rdp('prot', [p], shortcut=args.shortcut)
        print("Ensemble")
        evaluate_rdp('prot', emb_path, shortcut=args.shortcut)
