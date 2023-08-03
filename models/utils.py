import datetime

import esm
import torch
import yaml


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config(object):
    def __init__(self, config_file):
        self.config = yaml.load(open(config_file), Loader=yaml.FullLoader)

    def __getattr__(self, name):
        return self.config[name]

    def __str__(self):
        return str(self.config)


def EsmModelInfo(name: str):
    return {
        'esm2_t48_15B_UR50D': {'dim': 5120, 'layers': 48, 'model': esm.pretrained.esm2_t48_15B_UR50D},
        'esm2_t36_3B_UR50D': {'dim': 2560, 'layers': 36, 'model': esm.pretrained.esm2_t36_3B_UR50D},
        'esm2_t33_650M_UR50D': {'dim': 1280, 'layers': 33, 'model': esm.pretrained.esm2_t33_650M_UR50D},
        'esm2_t30_150M_UR50D': {'dim': 640, 'layers': 30, 'model': esm.pretrained.esm2_t30_150M_UR50D},
        'esm2_t12_35M_UR50D': {'dim': 480, 'layers': 12, 'model': esm.pretrained.esm2_t12_35M_UR50D},
        'esm2_t6_8M_UR50D': {'dim': 320, 'layers': 6, 'model': esm.pretrained.esm2_t6_8M_UR50D},
        'esm1b_t33_650M_UR50S': {'dim': 1280, 'layers': 33, 'model': esm.pretrained.esm1b_t33_650M_UR50S},
        'prot_t5_xl_half_uniref50-enc': {'dim': 1024, 'layers': 24, 'model': 'Rostlab/prot_t5_xl_uniref50'},
        'prot_t5_xl_bfd': {'dim': 1024, 'layers': 24, 'model': 'Rostlab/prot_t5_xl_bfd'},
        'ProtGPT2': {'dim': 1280, 'layers': 36, 'model': ''},
    }[name]


def write_log(results, args, output_path):
    with open(output_path, 'a') as f:
        f.write('-' * 50 + '\n')
        f.write(str(datetime.datetime.now()) + '\n')
        f.write(str(args) + '\n')
        f.write(str(results) + '\n')


RAAP = {  # relative amino acid propensities
    'A': [-0.40, -0.25, -0.08, -0.54, -0.40],
    'R': [1.4, 1.33, 0.12, 2.06, 1.87],
    'N': [0.09, -0.10, -0.15, -0.09, -0.22],
    'D': [-0.63, -0.62, -0.33, -0.67, -0.64],
    'C': [0.06, 0.17, 0.76, 0.10, -0.03],
    'Q': [-0.05, -0.17, -0.11, -0.00, -0.17],
    'E': [-0.70, -0.64, -0.34, -0.58, -0.61],
    'G': [-0.17, 0.04, -0.25, -0.41, -0.26],
    'H': [0.52, 0.54, 0.18, 0.51, 0.64],
    'I': [-0.09, 0.11, 0.71, -0.09, 0.16],
    'L': [-0.40, -0.18, 0.61, -0.35, -0.10],
    'K': [0.49, 0.51, -0.38, 0.36, 0.55],
    'M': [0.40, 0.55, 0.92, 0.36, 0.55],
    'F': [0.51, 0.41, 1.18, 0.64, 0.48],
    'P': [-0.46, -0.26, -0.17, -0.44, -0.23],
    'S': [0.23, -0.14, -0.13, 0.04, -0.40],
    'T': [0.12, -0.10, -0.07, 0.21, -0.17],
    'W': [1.14, 0.30, 0.95, 0.95, 0.77],
    'Y': [0.92, 0.33, 0.71, 0.71, 0.67],
    'V': [-0.27, 0.06, 0.37, -0.24, -0.10],
    'X': [0.0, 0.0, 0.0, 0.0, 0.0],
}
