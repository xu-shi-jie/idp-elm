from pathlib import Path

from utils import Config


def convert_config(p):
    config = Config(p)
    for k in [
        'esm1b_t33_650M_UR50S',
        'esm2_t6_8M_UR50D',
        'esm2_t12_35M_UR50D',
        'esm2_t30_150M_UR50D',
        'esm2_t33_650M_UR50D',
        'esm2_t36_3B_UR50D',
        'esm2_t48_15B_UR50D',
        'prot_t5_xl_bfd',
        'prot_t5_xl_half_uniref50-enc'
    ]:
        with open('configs/' + k + '.yaml', 'w') as f:
            for attr in config.__getattr__(k):
                f.write(attr + ': ' + str(config.__getattr__(k)[attr]) + '\n')
            if hasattr(config, 'shortcut'):
                f.write(f'shortcut: {config.shortcut}\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='')
    args = parser.parse_args()

    convert_config(args.config)
