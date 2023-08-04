import numpy as np
import pytorch_lightning as L
import torch
from dataloader import (ApodDataModule, CaidDataModule, DeepDisoBindDataModule,
                        DflDataModule, DisoLipPredDataModule, Dm4229DataModule,
                        NetSurfP30DataModule, UnionIdpDataModule,
                        flDPnnDataModule)
from nets import IdpPredictor, LinkerPredictor, RdpPredictor, SsPredictor
from utils import Config, seed_everything, write_log

import wandb

torch.set_float32_matmul_precision('high')


def get_callbacks(args: Config, tune=True, monitor='val_auc'):
    callbacks = [
        L.callbacks.ModelCheckpoint(
            dirpath='checkpoints',
            filename='{epoch}-{val_auc:.4f}' if tune else args.name,
            save_top_k=1,
            monitor=monitor,
            mode='max',
        ),
        L.callbacks.EarlyStopping(
            monitor=monitor,
            patience=args.patience,
            mode='max',
        ),
    ]
    return callbacks


def train_idp():
    wandb.init(project='idp-plm', )
    args = wandb.config
    args.add_dim = 8 if args.shortcut else 0
    seed_everything(args.seed)

    fldpnn = flDPnnDataModule(batch_size=args.batch_size, num_workers=args.num_workers, seed=args.seed)
    # dm = UnionIdpDataModule(batch_size=args.batch_size, num_workers=args.num_workers)
    model = IdpPredictor(args)
    if args.shortcut:
        model._load_auxiliary(Config('configs/ensemble_ss.yaml'))

    trainer = L.Trainer(accelerator='gpu', devices=1, max_epochs=args.epochs,
                        callbacks=get_callbacks(args), precision=16)

    # trainer.fit(model, train_dataloaders=dm.train_dataloader(open('data/Train/exclude.fldpnn.txt').readlines()),
    #             val_dataloaders=dm.val_dataloader(open('data/Train/exclude.fldpnn.txt').readlines()))
    trainer.fit(model, train_dataloaders=fldpnn.train_dataloader(),
                val_dataloaders=fldpnn.val_dataloader())

    # r = trainer.validate(model, dataloaders=dm.val_dataloader(), ckpt_path='best')
    r1 = trainer.validate(model, dataloaders=fldpnn.test_dataloader(), ckpt_path='best')
    # r2 = trainer.validate(model, dataloaders=caid.test_dataloader(), ckpt_path='best')

    # add test prefix
    r1 = {k.replace('val', 'test'): v for k, v in r1[0].items()}
    # r2 = {k.replace('val', 'caid'): v for k, v in r2[0].items()}
    # r = {**r[0], **r1, **r2}
    wandb.log(r1)


def train_idp_(config_path, test_name):
    args = Config(config_path)
    args.add_dim = 8 if args.shortcut else 0
    seed_everything(args.seed)
    # dm = UnionIdpDataModule(batch_size=args.batch_size, num_workers=args.num_workers)
    dm = flDPnnDataModule(batch_size=args.batch_size, num_workers=args.num_workers, seed=args.seed)
    model = IdpPredictor(args)
    if args.shortcut:
        model._load_auxiliary(Config('configs/ensemble_ss.yaml'))

    trainer = L.Trainer(accelerator='gpu', devices=1, max_epochs=args.epochs,
                        callbacks=get_callbacks(args, tune=False), precision=16)

    # exclude_file = f'data/Train/exclude.{test_name}.txt'
    # exclude_ids = open(exclude_file).read().splitlines()
    # trainer.fit(model, train_dataloaders=dm.train_dataloader(
    #     exclude_ids), val_dataloaders=dm.val_dataloader(exclude_ids))
    trainer.fit(model, train_dataloaders=dm.train_dataloader(),
                val_dataloaders=dm.val_dataloader())


def train_linker():
    wandb.init(project='idp-plm-search',)
    args = wandb.config
    # read sweep

    print(args)
    seed_everything(args.seed)
    dm = ApodDataModule(batch_size=args.batch_size, num_workers=args.num_workers)
    model = LinkerPredictor(args)
    model._load_auxiliary(Config('configs/ensemble_idp.yaml'), Config('configs/ensemble_ss.yaml'))

    trainer = L.Trainer(accelerator='gpu', devices=1, max_epochs=args.epochs,
                        callbacks=get_callbacks(args), precision=16)
    trainer.fit(model, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())
    r1 = trainer.validate(model, dataloaders=dm.val_dataloader(), ckpt_path='best')
    te64 = trainer.validate(model, dataloaders=dm.test_dataloader('te64'), ckpt_path='best')
    te82 = trainer.validate(model, dataloaders=dm.test_dataloader('te82'), ckpt_path='best')

    # add test prefix
    r64 = {k.replace('val', 'te64'): v for k, v in te64[0].items()}
    r82 = {k.replace('val', 'te82'): v for k, v in te82[0].items()}

    wandb.log({**r1[0], **r64, **r82})


def train_linker_(p):
    args = Config(p)
    print(args)
    seed_everything(args.seed)
    dm = ApodDataModule(batch_size=args.batch_size, num_workers=args.num_workers)
    model = LinkerPredictor(args)
    model._load_auxiliary(Config('configs/ensemble_idp.yaml'), Config('configs/ensemble_ss.yaml'))
    trainer = L.Trainer(accelerator='gpu', devices=1, max_epochs=args.epochs,
                        callbacks=get_callbacks(args, tune=False), precision=16)
    trainer.fit(model, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())
    r = trainer.validate(model, dataloaders=dm.test_dataloader('te64'), ckpt_path='best')
    write_log(r, args, 'models/results.txt')


def train_rdp(task: str):
    wandb.init(project='idp-plm-search',)
    args = wandb.config
    args.task = task
    print(args)

    seed_everything(args.seed)

    dm = DeepDisoBindDataModule(batch_size=args.batch_size, num_workers=args.num_workers)
    model = RdpPredictor(args)
    model._load_auxiliary(Config('configs/ensemble_idp.yaml'), Config('configs/ensemble_ss.yaml'))

    trainer = L.Trainer(accelerator='gpu', devices=1, max_epochs=args.epochs,
                        callbacks=get_callbacks(args, monitor='val_f1'), precision=16)
    trainer.fit(model, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())
    r1 = trainer.validate(model, dataloaders=dm.val_dataloader(), ckpt_path='best')
    r2 = trainer.validate(model, dataloaders=dm.test_dataloader(), ckpt_path='best')
    r2 = {k.replace('val', 'test'): v for k, v in r2[0].items()}
    wandb.log({**r1[0], **r2})


def train_rdp_(task: str, p):
    args = Config(p)
    args.task = task
    seed_everything(args.seed)
    dm = DeepDisoBindDataModule(batch_size=args.batch_size, num_workers=args.num_workers)
    model = RdpPredictor(args)
    model._load_auxiliary(Config('configs/ensemble_idp.yaml'), Config('configs/ensemble_ss.yaml'))
    trainer = L.Trainer(accelerator='gpu', devices=1, max_epochs=args.epochs,
                        callbacks=get_callbacks(args, tune=False), precision=16)
    trainer.fit(model, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())
    r = trainer.validate(model, dataloaders=dm.val_dataloader(), ckpt_path='best')
    write_log(r, args, 'models/results.txt')


def train_ss():
    wandb.init(project='idp-plm-search',)
    args = wandb.config
    print(args)

    seed_everything(args.seed)

    dm = NetSurfP30DataModule(batch_size=args.batch_size, num_workers=args.num_workers)
    model = SsPredictor(args)

    trainer = L.Trainer(accelerator='gpu', devices=1, max_epochs=args.epochs,
                        callbacks=get_callbacks(args), precision=16)
    trainer.fit(model, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())
    r1 = trainer.validate(model, dataloaders=dm.val_dataloader(), ckpt_path='best')
    casp12 = trainer.validate(model, dataloaders=dm.test_dataloader('casp12'), ckpt_path='best')
    cb513 = trainer.validate(model, dataloaders=dm.test_dataloader('cb513'), ckpt_path='best')
    ts115 = trainer.validate(model, dataloaders=dm.test_dataloader('ts115'), ckpt_path='best')
    casp12 = {k.replace('val', 'casp12'): v for k, v in casp12[0].items()}
    cb513 = {k.replace('val', 'cb513'): v for k, v in cb513[0].items()}
    ts115 = {k.replace('val', 'ts115'): v for k, v in ts115[0].items()}
    wandb.log({**r1[0], **casp12, **cb513, **ts115})


def train_ss_(p):
    args = Config(p)
    print(args)

    seed_everything(args.seed)
    dm = NetSurfP30DataModule(batch_size=args.batch_size, num_workers=args.num_workers)
    model = SsPredictor(args)

    trainer = L.Trainer(accelerator='gpu', devices=1, max_epochs=args.epochs,
                        callbacks=get_callbacks(args, tune=False), precision=16)
    trainer.fit(model, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())
    r = trainer.validate(model, dataloaders=dm.val_dataloader(), ckpt_path='best')
    write_log(r, args, 'models/results.txt')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='idp')
    parser.add_argument('--path', type=str, default='configs/esm1b_t33_650M_UR50S.yaml')
    parser.add_argument('--test', type=str, default='fldpnn')
    args = parser.parse_args()

    if args.task == 'ss':
        train_ss_(args.path)
    elif args.task == 'idp':
        train_idp_(args.path, args.test)
    elif args.task == 'linker':
        train_linker_(args.path)
    elif args.task == 'prot':
        train_rdp_(args.task, args.path)

    # train_ss()
    # train_idp()
    # train_linker()
    # train_rdp('prot')
