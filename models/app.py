from pathlib import Path

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
    args = Config('configs/ensemble_idp.yaml')
    args.add_dim = 0
    seed_everything(args.seed)

    ver = 'v23.4.13'

    idp_plm = EnsemblePredictor(args).to('cuda:1')
    model_names = [
        'facebook/esm1b_t33_650M_UR50S',
        'facebook/esm2_t6_8M_UR50D',
        'facebook/esm2_t12_35M_UR50D',
        'facebook/esm2_t30_150M_UR50D',
        'facebook/esm2_t33_650M_UR50D',
        'facebook/esm2_t36_3B_UR50D',
        'Rostlab/prot_t5_xl_bfd',
        'Rostlab/prot_t5_xl_half_uniref50-enc',
    ]

    @st.cache(allow_output_mutation=True)
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
        print("Done loading encoders")
        return encoders

    st.title('IDP-PLM' + ver)
    # click to copy to clipboard button
    st.markdown('Example sequence (**SARS Covid Nucleocapsid**, click to copy):')
    st.code('''MSDNGPQNQRNAPRITFGGPSDSTGSNQNGERSGARSKQRRPQGLPNNTASWFTALTQHGKEDLKFPRGQGVPINTNSSPDDQIGYYRRATRRIRGGDGKMKDLSPRWYFYYLGTGPEAGLPYGANKDGIIWVATEGALNTPKDHIGTRNPANNAAIVLQLPQGTTLPKGFAEGSRGGSQASSRSSSRSRNSSRNSTPGSSRGTSPARMAGNGGDAALALLLLDRLNQLESKMSGKGQQQQGQTVTKKSAAEASKKPRQKRTATKAYNVTQAFGRRGPEQTQGNFGDQELIRQGTDYKHWPQIAQFAPSASAFFGMSRIGMEVTPSGTWLTYTGAIKLDDKDPNFKQVILLNKHIDAYKTFPPTEPKKDKKKKADETQALPQRQKKQQTVTLLPAADLDDFSKQLQQSMSSADSTQA''', language='text')

    seq = st.text_area('**Input sequence**', placeholder='Sequence', height=200)

    if seq != '':
        encoders = get_encoders()
        embs = {}
        for name, model in encoders.items():
            emb = model([seq])
            embs[name] = emb.to('cuda:0').detach()

        logits = idp_plm(embs)

        pred = logits.sum(-1).softmax(-1)[..., 1].detach().cpu().numpy()

        full_dis = (logits.argmax(-2).sum(-2) / len(seq)).mean() > 0.95

        st.header('Results')
        st.write('Sequence: ', seq)
        st.write('Length: ', len(seq))
        st.markdown(f'##### Full disorder: {full_dis.item()}')

        st.plotly_chart({
            'data': [
                {'x': list(range(1, 1 + len(seq))), 'y': pred, 'type': 'scatter',
                 'name': 'Predictions', 'line': {'color': 'green', 'width': 2}},
                # draw average 0.5 line
                {'x': list(range(1, 1 + len(seq))),
                 'y': [0.5] * len(seq), 'type': 'scatter', 'name': 'Average', 'line': {'color': 'red', 'width': 1}},
            ],
            'layout': {
                'title_text': "IDP-PLM" + ver + " predictions",
                'xaxis_title': "Position",
                'yaxis_title': "Score",
                'font': dict(
                    family="Arial",
                    size=18,
                    color="black"),
                'xaxis': dict(showgrid=False),
                'yaxis': dict(showgrid=False),
                'yaxis_range': [0, 1],
                'legend': dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                )
            }
        })

        # clean up
        del embs
        del logits
        del pred
