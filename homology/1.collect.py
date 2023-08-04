# dm4229_train_data = open('data/DM4229/train.txt').readlines()
# dm4229_val_data = open('data/DM4229/val.txt').readlines()
fldpnn_train_data = open('data/flDPnn/flDPnn_Training_Annotation.txt').read().splitlines()[10:]
fldpnn_val_data = open('data/flDPnn/flDPnn_Validation_Annotation.txt').read().splitlines()[10:]
# espritz_data = open('data/ESpritz/train.fasta').readlines()
# test_data = open('data/flDPnn/flDPnn_DissimiTest_Annotation.txt').readlines()[10:]
# with open('tmp/in.txt', 'w') as f:
#     for i in range(0, len(train_data), 3):
#         f.write(f'{train_data[i].strip()}\n{train_data[i+1].strip()}\n')
#     for i in range(0, len(val_data), 3):
#         f.write(f'{val_data[i].strip()}\n{val_data[i+1].strip()}\n')
#     for i in range(0, len(test_data), 7):
#         f.write(f'{test_data[i].strip()}\n{test_data[i+1].strip()}\n')

# test_data = open('data/Test/CASP_clean.txt').readlines()
# test_data = open('data/Test/DISORDER723_clean.txt').readlines()
# test_data = open('data/Test/MXD494_clean.txt').readlines()
# test_data = open('data/Test/SL329_clean.txt').readlines()
# test_data = open('data/DisProt.txt').readlines()
test_data = open('data/CAID.txt').readlines()
# test_data = open('data/CAID2/disorder_pdb_2.txt').read().splitlines()
with open('tmp/in.txt', 'w') as f:
    for i in range(0, len(fldpnn_train_data), 7):
        f.write(f'{fldpnn_train_data[i].strip()}\n{fldpnn_train_data[i+1].strip()}\n')
    for i in range(0, len(fldpnn_val_data), 7):
        f.write(f'{fldpnn_val_data[i].strip()}\n{fldpnn_val_data[i+1].strip()}\n')
    # for i in range(0, len(dm4229_train_data), 3):
    #     f.write(f'{dm4229_train_data[i].strip()}\n{dm4229_train_data[i+1].strip()}\n')
    # for i in range(0, len(dm4229_val_data), 3):
    #     f.write(f'{dm4229_val_data[i].strip()}\n{dm4229_val_data[i+1].strip()}\n')
    # for i in range(0, len(espritz_data), 2):
    #     f.write(f'{espritz_data[i].strip()}\n{espritz_data[i+1].strip()}\n')
    for i in range(0, len(test_data), 3):
        f.write(f'{test_data[i].strip()}\n{test_data[i+1].strip()}\n')
