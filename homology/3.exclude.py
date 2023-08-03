
import re

from tqdm import tqdm

pat = re.compile(r'.+?>(.+?)\.\.\.')
pat2 = re.compile(r'.+?>(.+?)\.\.\..+?(\d+\.?\d+%)')
p = 'tmp/db25.txt.clstr'
with open(p) as f:
    cluster = {}
    for line in tqdm(f):
        if line.startswith('>'):
            cluster_id = line.strip()
        elif line[0] == '0':
            cluster[cluster_id] = [pat.findall(line)[0]]
        else:
            try:
                cluster[cluster_id] = cluster[cluster_id] + [pat2.findall(line)[0]]
            except:
                cluster[cluster_id] = cluster[cluster_id] + [pat.findall(line)[0]]

# test_data = open('data/flDPnn/flDPnn_DissimiTest_Annotation.txt').read().splitlines()[10:]
# test_ids = [s[1:] for s in test_data[::7]]
# test_data = open('data/Test/CASP_clean.txt').read().splitlines()
# test_data = open('data/Test/DISORDER723_clean.txt').read().splitlines()
# test_data = open('data/Test/MXD494_clean.txt').read().splitlines()
# test_data = open('data/Test/SL329_clean.txt').read().splitlines()
# test_data = open('data/DisProt.txt').read().splitlines()
# test_data = open('data/CAID2/disorder_pdb_2.txt').read().splitlines()
test_data = open('data/CAID.txt').read().splitlines()
test_ids = [s[1:] for s in test_data[::3]]

train_data = open('data/flDPnn/flDPnn_Training_Annotation.txt').read().splitlines()[10:]
val_data = open('data/flDPnn/flDPnn_Validation_Annotation.txt').read().splitlines()[10:]
train_ids = [s[1:] for s in train_data[::7]]
val_ids = [s[1:] for s in val_data[::7]]
# dm4229_train_data = open('data/DM4229/train.txt').read().splitlines()
# dm4229_train_ids = [s[1:] for s in dm4229_train_data[::3]]
# dm4229_val_data = open('data/DM4229/val.txt').read().splitlines()
# dm4229_val_ids = [s[1:] for s in dm4229_val_data[::3]]
# espritz_train_data = open('data/ESpritz/train.fasta').read().splitlines()
# espritz_train_ids = [s[1:] for s in espritz_train_data[::2]]
train_ids = train_ids + val_ids  # + dm4229_train_ids + dm4229_val_ids + espritz_train_ids

ids = []
for c in cluster.values():
    if len(c) == 1:
        ids.append(c)
    else:
        if c[0] in train_ids:
            continue
        else:
            for cc in c[1:]:
                if cc[0] in train_ids and float(cc[1][:-1]) > 25.0:
                    break
            else:
                ids.append(c[0])
                # for cc in c[1:]:
                #     if float(cc[1][:-1]) < 25.0:  # even the most similar representative is less than 25.0, so we can add it
                #         ids.append(cc[0])

# with open('tmp/exclude.txt', 'w') as f:
#     for e in exclude:
#         f.write(e + '\n')

ori_len = len(test_data) // 3
remain_len = 0
# with open('data/CAID2_clean.txt', 'w') as f:
with open('data/CAID_clean.txt', 'w') as f:
    for i in range(0, len(test_data), 3):
        if test_data[i][1:] in ids:
            f.write(test_data[i] + '\n')
            f.write(test_data[i + 1] + '\n')
            f.write(test_data[i + 2] + '\n')
            remain_len += 1

print(f'Original length: {ori_len}, remain length: {remain_len}')
