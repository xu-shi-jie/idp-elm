# Author: Shijie Xu
# Date: 2023-03-19
# Description: ESM encoder

import itertools
from pathlib import Path

import loguru
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import infer_auto_device_map, init_empty_weights
from dataloader import (ApodDataModule, Caid2DataModule, CaidDataModule,
                        DeepDisoBindDataModule, DisoLipPredDataModule,
                        DisProt2022DecDataModule, Dm4229DataModule,
                        NetSurfP30DataModule, VariousIdpTestDataModule,
                        flDPnnDataModule)
from loguru import logger
from tqdm import tqdm
from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          T5EncoderModel, T5Model, T5Tokenizer)
from utils import seed_everything


class ProtEncoder(nn.Module):
    def __init__(self, model_name, dev):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name, device_map="balanced", torch_dtype=torch.float16, offload_folder="offload",  # auto, balanced_low_0
            offload_state_dict=True,)
        self.max_len = 960
        self.overlap = 31
        # self.model.to(dev)
        self.model.eval()

    def forward(self, _seqs):
        with torch.no_grad() and torch.cuda.amp.autocast():
            assert len(_seqs) == 1, 'Only support batch size 1'

            seqs = _seqs[0]

            # left overlappping, right overlappping
            seqs = [seqs[max(0, i - self.overlap):(i + self.max_len + self.overlap)]
                    for i in range(0, len(seqs), self.max_len)]

            segs = []
            for seq in seqs:
                inputs = self.tokenizer([seq], return_tensors="pt").to(self.model.device)
                outputs = self.model(**inputs).last_hidden_state.squeeze(0).detach().cpu()
                segs.append(outputs)

            t = []
            for i in range(len(seqs)):
                if i == 0:
                    t.append(segs[i][1:(1 + self.max_len)])
                elif i == len(seqs) - 1:
                    t.append(segs[i][1 + self.overlap:])
                else:
                    t.append(segs[i][1 + self.overlap:1 + self.max_len + self.overlap])

            outputs = torch.cat(t, dim=0)[:len(_seqs[0])]
            assert outputs.shape[0] == len(_seqs[0])
            return outputs


class T5Encoder(nn.Module):
    def __init__(self, name: str, dev) -> None:
        super().__init__()
        self.dev = dev
        if name == 'Rostlab/prot_t5_xl_half_uniref50-enc':
            # Load the tokenizer
            self.tokenizer = T5Tokenizer.from_pretrained(
                'Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False,
            )
            # Load the model
            self.model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(dev)
        elif name == 'Rostlab/prot_t5_xl_bfd':
            # Load the tokenizer
            self.tokenizer = T5Tokenizer.from_pretrained(
                'Rostlab/prot_t5_xl_bfd', do_lower_case=False,
            )
            # Load the model
            self.model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_bfd").to(dev)
        self.max_len = 960  # start_token, end_token occupy 2 positions
        self.overlap = 31

    def forward(self, _seqs):
        with torch.no_grad():
            assert len(_seqs) == 1, 'Only support batch size 1'

            seqs = _seqs[0]

            # left overlappping, right overlappping
            seqs = [seqs[max(0, i - self.overlap):(i + self.max_len + self.overlap)]
                    for i in range(0, len(seqs), self.max_len)]
            # seqs = [seqs[i:i + self.max_len] for i in range(0, len(seqs), self.max_len)] # no sliding window

            input_ids = self.tokenizer.batch_encode_plus(
                [' '.join(list(s)) for s in seqs],
                add_special_tokens=True, padding="longest")['input_ids']
            input_ids = torch.tensor(input_ids).to(self.dev)

            outputs = self.model(input_ids=input_ids)

            outputs = outputs.last_hidden_state

            t = []
            for i in range(len(seqs)):
                if i == 0:
                    t.append(outputs[i, 1:(1 + self.max_len)])
                elif i == len(seqs) - 1:
                    t.append(outputs[i, 1 + self.overlap:])
                else:
                    t.append(outputs[i, 1 + self.overlap:1 + self.max_len + self.overlap])

            outputs = torch.cat(t, dim=0)[:len(_seqs[0])]
            assert outputs.shape[0] == len(_seqs[0])
            return outputs


def gen_embs(task: str):
    plms = [
        'facebook/esm2_t6_8M_UR50D',
        'facebook/esm2_t12_35M_UR50D',
        'facebook/esm2_t30_150M_UR50D',
        'facebook/esm2_t33_650M_UR50D',
        'facebook/esm1b_t33_650M_UR50S',
        'Rostlab/prot_t5_xl_bfd',
        'Rostlab/prot_t5_xl_half_uniref50-enc',
        'facebook/esm2_t36_3B_UR50D',
        'facebook/esm2_t48_15B_UR50D',
    ]

    for plm_idx, model_name in enumerate(plms):
        emb_path = Path(f'embs/{task}/{model_name}')
        emb_path.mkdir(parents=True, exist_ok=True)

        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        encoder = T5Encoder(model_name, dev) if model_name.startswith('Rostlab') else ProtEncoder(model_name, dev)
        match task:
            case 'idp':
                logger.info(f'[{plm_idx+1}/{len(plms)}] Performing PLM {model_name} with {dev} on flDPnn/caid/caid2 datasets, saving to {emb_path}')

                fldpnn = flDPnnDataModule(batch_size=1, num_workers=1, seed=42)
                caid = CaidDataModule(batch_size=1, num_workers=1)
                caid2 = Caid2DataModule(batch_size=1, num_workers=1)
                t = itertools.chain(
                    fldpnn.train_dataloader(),
                    fldpnn.val_dataloader(),
                    fldpnn.test_dataloader(),
                    caid.test_dataloader(),
                    caid2.test_dataloader(),
                )
            case 'linker':
                logger.info(f'[{plm_idx+1}/{len(plms)}] Performing PLM {model_name} with {dev} on dfl dataset, saving to {emb_path}')

                dm = ApodDataModule(batch_size=1, num_workers=1)
                t = itertools.chain(
                    dm.train_dataloader(),
                    dm.val_dataloader(),
                    dm.test_dataloader('te82'),
                    dm.test_dataloader('te64'))
                
            case 'rdp':
                logger.info(f'[{plm_idx+1}/{len(plms)}] Performing PLM {model_name} with {dev} on DeepDISOBind datasets, saving to {emb_path}')

                dm = DeepDisoBindDataModule(batch_size=1, num_workers=1)
                t = itertools.chain(dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader())

            case 'ss':
                logger.info(f'[{plm_idx+1}/{len(plms)}] Performing PLM {model_name} with {dev} on NetSurfP-3.0 datasets, saving to {emb_path}')

                dm = NetSurfP30DataModule(batch_size=1, num_workers=1)
                t = itertools.chain(
                    dm.train_dataloader(),
                    dm.val_dataloader(),
                    dm.test_dataloader('casp12'),
                    dm.test_dataloader('cb513'),
                    dm.test_dataloader('ts115'))

        for batch in tqdm(t, leave=False):
            for id, seq, *_ in batch:
                if (emb_path / f'{id[1:]}.pt').exists():
                    continue
                emb = encoder([seq])
                torch.save(emb, emb_path / f'{id[1:]}.pt')


if __name__ == '__main__':
    seed_everything(42)
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument('--task', type=str, default='idp')
    args = args.parse_args()

    gen_embs(args.task)
