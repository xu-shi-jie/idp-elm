import numpy as np
import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from utils import RAAP, Config, EsmModelInfo


class Head(nn.Module):
    def __init__(self, type, input_size, hidden_size, num_layers, dropout, output_size=2) -> None:
        super().__init__()
        if type in ['bilstm', 'lstm']:
            self.net = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=(type == 'bilstm'),
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )
        elif type in ['bigru', 'gru']:
            self.net = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=(type == 'bigru'),
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )

        self.dropout = nn.Dropout(dropout)

        if type in ['bilstm', 'bigru']:
            self.fc = nn.Linear(hidden_size * 2, output_size)
        else:
            self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.net(x)
        return self.fc(self.dropout(x))


class IdpPredictor(L.LightningModule):
    def __init__(self, args: Config):
        super().__init__()
        self.args = args

        self.model_name = args.emb_dir.split('/')[-1]
        self.esm_info = EsmModelInfo(self.model_name)

        self.head = Head(
            type=args.head_type,
            input_size=self.esm_info['dim'] + args.add_dim,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _load_auxiliary(self, ss_config: str):
        ss_head = Head(
            type=ss_config.__getattr__(self.model_name)['head_type'],
            input_size=self.esm_info['dim'],
            hidden_size=ss_config.__getattr__(self.model_name)['hidden_size'],
            num_layers=ss_config.__getattr__(self.model_name)['num_layers'],
            dropout=ss_config.__getattr__(self.model_name)['dropout'],
            output_size=8,
        )
        p = f'weights/ss/' + self.model_name + '.ckpt'
        state_dict = torch.load(p)['state_dict']
        # replace 'head.' with ''
        state_dict = {k.replace('head.', ''): v for k, v in state_dict.items()}
        ss_head.load_state_dict(state_dict)
        ss_head.to('cuda')
        ss_head.eval()

        # do not set ss_head to a submodel of self, because it will be saved in the checkpoint
        self.ss_head = [ss_head]

    def configure_optimizers(self):
        self.loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=float(self.args.weight_decay))
        return optimizer

    def forward(self, x):
        if self.args.shortcut:
            with torch.no_grad():
                ss = self.ss_head[0](x).detach()
            x = torch.cat([x, ss], dim=-1)

        return self.head(x)

    def step(self, batch):
        # read emb
        ids, _, dis = zip(*batch)
        embs = [torch.load(f'{self.args.emb_dir}/{id[1:]}.pt').to(self.dev) for id in ids]
        for i in range(len(embs)):
            assert len(embs[i]) == len(dis[i]), f'{len(embs[i])} != {len(dis[i])}'
        # dis to tensor
        # pad emb
        embs = torch.nn.utils.rnn.pad_sequence(embs, batch_first=True)
        # forward
        logits = self.forward(embs)
        # flatten logits
        logits = torch.cat([logits[i, :len(dis[i])] for i in range(len(dis))])
        # dis: str to tensor
        dis = torch.cat([torch.tensor([int(d) for d in di]) for di in dis]).to(self.dev)

        if self.training:
            available_mask = (dis == 0).bool()  # where we can switch the label
            mask = (dis == 1).bool()
            mask |= (available_mask & (torch.rand(*dis.shape).to(self.dev) < float(self.args.mask_ratio)))
        else:
            mask = torch.ones_like(dis).bool()  # this is no mask in fldpnn dataset !!

        return logits[mask], dis[mask]

    def training_step(self, batch, batch_idx):
        y_hat, y = self.step(batch)
        loss = self.loss(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            y_hat, y = self.step(batch)
            self.y_hats.append(y_hat)
            self.ys.append(y)

    def on_validation_epoch_start(self) -> None:
        self.y_hats = []
        self.ys = []

    def on_validation_epoch_end(self) -> None:
        y_hat = torch.cat(self.y_hats)
        y = torch.cat(self.ys)
        self.log('val_loss', self.loss(y_hat, y), prog_bar=True)
        y_hat = y_hat.softmax(-1)[:, 1]

        self.log('val_auc', auc := roc_auc_score(y.cpu(), y_hat.cpu()), prog_bar=True)
        self.log('val_f1', f1 := f1_score(y.cpu(), (y_hat > 0.5).cpu()))
        self.log('val_mcc', mcc := matthews_corrcoef(y.cpu(), (y_hat > 0.5).cpu()))
        self.log('val_pre', precision_score(y.cpu(), (y_hat > 0.5).cpu()))
        self.log('val_rec', recall_score(y.cpu(), (y_hat > 0.5).cpu()))
        self.log('val_stop', auc + f1 + mcc)


class LinkerPredictor(L.LightningModule):
    def __init__(self, args: Config):
        super().__init__()
        self.args = args

        self.model_name = args.emb_dir.split('/')[-1]
        self.esm_info = EsmModelInfo(self.model_name)

        self.head = Head(
            type=args.head_type,
            input_size=self.esm_info['dim'] + (8 if args.shortcut else 0),
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )

        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.token = {'1': 1, '0': 0, 'x': 2}

    def _load_auxiliary(self, idp_config: str, ss_config: str):
        idp_head = Head(
            type=idp_config.__getattr__(self.model_name)['head_type'],
            input_size=self.esm_info['dim'],
            hidden_size=idp_config.__getattr__(self.model_name)['hidden_size'],
            num_layers=idp_config.__getattr__(self.model_name)['num_layers'],
            dropout=idp_config.__getattr__(self.model_name)['dropout'],
            output_size=2,
        )
        p = f'weights/idp/' + self.model_name + '.ckpt'
        state_dict = torch.load(p)['state_dict']
        # replace 'head.' with ''
        state_dict = {k.replace('head.', ''): v for k, v in state_dict.items()}
        idp_head.load_state_dict(state_dict)
        idp_head.to('cuda')
        idp_head.eval()
        self.idp_head = [idp_head]

        if self.args.shortcut:
            ss_head = Head(
                type=ss_config.__getattr__(self.model_name)['head_type'],
                input_size=self.esm_info['dim'],
                hidden_size=ss_config.__getattr__(self.model_name)['hidden_size'],
                num_layers=ss_config.__getattr__(self.model_name)['num_layers'],
                dropout=ss_config.__getattr__(self.model_name)['dropout'],
                output_size=8,
            )
            p = f'weights/ss/' + self.model_name + '.ckpt'
            state_dict = torch.load(p)['state_dict']
            # replace 'head.' with ''
            state_dict = {k.replace('head.', ''): v for k, v in state_dict.items()}
            ss_head.load_state_dict(state_dict)
            ss_head.to('cuda')
            ss_head.eval()

            # do not set ss_head and idp_head to a submodel of self, because it will be saved in the checkpoint
            self.ss_head = [ss_head]

    def configure_optimizers(self):
        self.loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=float(self.args.weight_decay))
        return optimizer

    def forward(self, x):
        # idp = self.idp_head[0](x).detach()
        if self.args.shortcut:
            ss = self.ss_head[0](x).detach()
            x = torch.cat([x, ss], dim=-1)
        # else:
        #     x = torch.cat([x, idp], dim=-1)

        return self.head(x)

    def step(self, batch):
        ids, _, label = zip(*batch)
        embs = [torch.load(f'{self.args.emb_dir}/{id[1:]}.pt').to(self.dev) for id in ids]
        embs = torch.nn.utils.rnn.pad_sequence(embs, batch_first=True)
        logits = self.forward(embs)
        logits = torch.cat([logits[i, :len(label[i])] for i in range(len(label))])
        label = torch.cat([torch.tensor([self.token[l] for l in la]) for la in label]).to(self.dev)

        mask = (label != 2).bool()

        if self.training:
            available_mask = (label == 0).bool()  # where we can switch the label
            mask = (label == 1).bool()
            mask |= (available_mask & (torch.rand(*label.shape).to(self.dev) < float(self.args.mask_ratio)))
        else:
            # mask = torch.ones_like(label).bool()
            mask = (label != 2).bool()

        return logits[mask], label[mask]

    def training_step(self, batch, batch_idx):
        y_hat, y = self.step(batch)
        loss = self.loss(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            y_hat, y = self.step(batch)
            self.y_hats.append(y_hat)
            self.ys.append(y)

    def on_validation_epoch_start(self) -> None:
        self.y_hats = []
        self.ys = []

    def on_validation_epoch_end(self) -> None:
        y_hat = torch.cat(self.y_hats)
        y = torch.cat(self.ys)
        self.log('val_loss', self.loss(y_hat, y), prog_bar=True)
        y_hat = y_hat.softmax(-1)[:, 1]

        try:
            self.log('val_auc', auc := roc_auc_score(y.cpu(), y_hat.cpu()), prog_bar=True)
            self.log('val_f1', f1 := f1_score(y.cpu(), (y_hat > 0.5).cpu()))
            self.log('val_mcc', mcc := matthews_corrcoef(y.cpu(), (y_hat > 0.5).cpu()))
            self.log('val_pre', precision_score(y.cpu(), (y_hat > 0.5).cpu()))
            self.log('val_rec', recall_score(y.cpu(), (y_hat > 0.5).cpu()))
            self.log('val_stop', auc + f1 + mcc)
        except:
            pass


class RdpPredictor(L.LightningModule):
    def __init__(self, args: Config):
        super().__init__()
        self.args = args

        model_name = args.emb_dir.split('/')[-1]
        esm_info = EsmModelInfo(model_name)
        self.model_name = model_name
        self.esm_info = esm_info

        if args.task in ['prot', 'dna', 'rna']:  # BiRNN
            self.head = Head(
                type=args.head_type,
                input_size=esm_info['dim'] + (10 if args.shortcut else 2),
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                dropout=args.dropout,
                output_size=2,
            )

        self.task = args.task
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.token = {'0': 0, '1': 1}

    def _load_auxiliary(self, idp_config: str, ss_config: str):
        idp_head = Head(
            type=idp_config.__getattr__(self.model_name)['head_type'],
            input_size=self.esm_info['dim'],
            hidden_size=idp_config.__getattr__(self.model_name)['hidden_size'],
            num_layers=idp_config.__getattr__(self.model_name)['num_layers'],
            dropout=idp_config.__getattr__(self.model_name)['dropout'],
            output_size=2,
        )
        p = f'weights/idp/' + self.model_name + '.ckpt'
        state_dict = torch.load(p)['state_dict']
        # replace 'head.' with ''
        state_dict = {k.replace('head.', ''): v for k, v in state_dict.items()}
        idp_head.load_state_dict(state_dict)
        idp_head.to('cuda')
        idp_head.eval()
        self.idp_head = [idp_head]

        if self.args.shortcut:
            ss_head = Head(
                type=ss_config.__getattr__(self.model_name)['head_type'],
                input_size=self.esm_info['dim'],
                hidden_size=ss_config.__getattr__(self.model_name)['hidden_size'],
                num_layers=ss_config.__getattr__(self.model_name)['num_layers'],
                dropout=ss_config.__getattr__(self.model_name)['dropout'],
                output_size=8,
            )
            p = f'weights/ss/' + self.model_name + '.ckpt'
            state_dict = torch.load(p)['state_dict']
            # replace 'head.' with ''
            state_dict = {k.replace('head.', ''): v for k, v in state_dict.items()}
            ss_head.load_state_dict(state_dict)
            ss_head.to('cuda')
            ss_head.eval()

            # do not set ss_head and idp_head to a submodel of self, because it will be saved in the checkpoint
            self.ss_head = [ss_head]

    def configure_optimizers(self):
        self.loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=float(self.args.weight_decay))
        return optimizer

    def step(self, batch, joint=False):
        ids, _, dis, prot, dna, rna = zip(*batch)

        embs = [torch.load(f'{self.args.emb_dir}/{id[1:]}.pt').to(self.dev) for id in ids]
        embs = torch.nn.utils.rnn.pad_sequence(embs, batch_first=True)

        logits = self.forward(embs)
        logits = torch.cat([logits[i, :len(dis[i])] for i in range(len(dis))])

        label = prot if self.task == 'prot' else dna if self.task == 'dna' else rna
        label = torch.cat([torch.tensor([self.token[l] for l in la]) for la in label]).to(self.dev)
        if self.training:
            available_mask = (label == 0).bool()  # where we can switch the label
            mask = (label == 1).bool()
            mask |= (available_mask & (torch.rand(*label.shape).to(self.dev) < float(self.args.mask_ratio)))
        else:
            mask = torch.ones_like(label).bool()

        return logits[mask], label[mask]

    def forward(self, x):
        idp = self.idp_head[0](x).detach()

        if self.args.shortcut:
            ss = self.ss_head[0](x).detach()
            x = torch.cat([x, idp, ss], dim=-1)
        else:
            x = torch.cat([x, idp], dim=-1)

        return self.head(x)

    def training_step(self, batch, batch_idx):
        joint = False

        if joint:
            y_hat, label_prot, label_dna, label_rna = self.step(batch, joint=True)
            loss_prot = self.loss(y_hat[..., :2], label_prot)
            loss_dna = self.loss(y_hat[..., 2:4], label_dna)
            loss_rna = self.loss(y_hat[..., 4:], label_rna)
            return loss_prot + loss_dna + loss_rna
        else:
            y_hat, label = self.step(batch, joint=False)
            loss = self.loss(y_hat, label)
            return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            joint = False
            if joint:
                y_hat, label_prot, label_dna, label_rna = self.step(batch, joint=True)
                self.y_hats.append(y_hat)
                self.ys.append(torch.stack([label_prot, label_dna, label_rna], dim=-1))
            else:
                y_hat, label = self.step(batch, joint=False)
                self.y_hats.append(y_hat)
                self.ys.append(label)

    def on_validation_epoch_start(self) -> None:
        self.y_hats = []
        self.ys = []

    def on_validation_epoch_end(self) -> None:
        y_hats = torch.cat(self.y_hats)
        ys = torch.cat(self.ys)

        def log_metrics(y_hat, y, prefix):
            y_hat = y_hat.softmax(-1)[:, 1]

            threshold = np.linspace(0, 1, 100)
            specificity = []
            for t in threshold:
                specificity.append(((y_hat[y == 0] < t).sum() / (y == 0).sum()).item())
            # set threshold to the point where specificity is 0.8
            t = threshold[np.argmin(np.abs(np.array(specificity) - 0.8))]

            try:
                self.log(f'{prefix}auc', auc := roc_auc_score(y.cpu(), y_hat.cpu()), prog_bar=True)
                self.log(f'{prefix}f1', f1 := f1_score(y.cpu(), (y_hat > t).cpu()), prog_bar=True)
                self.log(f'{prefix}mcc', mcc := matthews_corrcoef(y.cpu(), (y_hat > t).cpu()), prog_bar=True)
                self.log(f'{prefix}pre', precision_score(y.cpu(), (y_hat > t).cpu()))
                self.log(f'{prefix}rec', recall_score(y.cpu(), (y_hat > t).cpu()))
                self.log(f'{prefix}stop', auc + f1 + mcc)
                return f1
            except:
                self.log(f'{prefix}auc', auc := 0.5, prog_bar=True)
                return 0.0

        joint = False
        if joint:
            prot_f1 = log_metrics(y_hats[..., :2], ys[..., 0], 'prot_')
            dna_f1 = log_metrics(y_hats[..., 2:4], ys[..., 1], 'dna_')
            rna_f1 = log_metrics(y_hats[..., 4:], ys[..., 2], 'rna_')

            self.log('val_auc', (prot_f1 + dna_f1 + rna_f1) / 3, prog_bar=True)
        else:
            log_metrics(y_hats, ys, 'val_')


class SsPredictor(L.LightningModule):
    def __init__(self, args: Config):
        super().__init__()
        self.args = args

        model_name = args.emb_dir.split('/')[-1]
        esm_info = EsmModelInfo(model_name)

        self.head = Head(
            type=args.head_type,
            input_size=esm_info['dim'],
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            output_size=8
        )

        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def configure_optimizers(self):
        self.loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=float(self.args.weight_decay))
        return optimizer

    def forward(self, x):
        return self.head(x)

    def step(self, batch):
        ids, _, label = zip(*batch)
        embs = [torch.load(f'{self.args.emb_dir}/{id[1:]}.pt').to(self.dev) for id in ids]
        for i in range(len(embs)):
            assert len(embs[i]) == len(label[i]), f'{ids[i]}: {len(embs[i])} != {len(label[i])}'
        embs = torch.nn.utils.rnn.pad_sequence(embs, batch_first=True)
        logits = self.forward(embs)
        logits = torch.cat([logits[i, :len(label[i])] for i in range(len(label))])
        label = torch.cat([torch.tensor([int(l) for l in la]) for la in label]).to(self.dev)
        mask = (label != 8).bool()
        return logits[mask], label[mask]

    def training_step(self, batch, batch_idx):
        y_hat, y = self.step(batch)
        loss = self.loss(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            y_hat, y = self.step(batch)
            self.y_hats.append(y_hat)
            self.ys.append(y)

    def on_validation_epoch_start(self) -> None:
        self.y_hats = []
        self.ys = []

    def on_validation_epoch_end(self) -> None:
        y_hats = torch.cat(self.y_hats)
        ys = torch.cat(self.ys)
        self.log(f'val_auc', accuracy_score(ys.cpu(), y_hats.argmax(-1).cpu()), prog_bar=True)
