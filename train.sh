#!/bin/bash
while true; do
    echo "=============================================================================="
    echo "Select a task to reproduce IDP-PLM results (recommended to run in order):"
    echo "[0] Encode sequences"
    echo "[1] Train secondary structure predictor"
    echo "  [2] Evaluate secondary structure predictor"
    echo "[3] Train intrinsically disordered protein predictor"
    echo "  [4] Train intrinsically disordered protein predictor (w/ ss)"
    echo "  [5] Evaluate intrinsically disordered protein predictor"
    echo "  [6] Evaluate intrinsically disordered protein predictor (w/ ss)"
    echo "[7] Train disordered flexible linker predictor"
    echo "  [8] Train disordered flexible linker predictor (w/ ss)"
    echo "  [9] Evaluate disordered flexible linker predictor"
    echo "  [10] Evaluate disordered flexible linker predictor (w/ ss)"
    echo "[11] Train disordered protein-binding predictor"
    echo "  [12] Train disordered protein-binding predictor (w/ ss)"
    echo "  [13] Evaluate disordered protein-binding predictor"
    echo "  [14] Evaluate disordered protein-binding predictor (w/ ss)"
    echo "[q] Quit"
    echo "=============================================================================="
    read ch

    declare -a emb_dirs=("configs/esm1b_t33_650M_UR50S.yaml" "configs/esm2_t6_8M_UR50D.yaml" "configs/esm2_t12_35M_UR50D.yaml" "configs/esm2_t30_150M_UR50D.yaml" "configs/esm2_t33_650M_UR50D.yaml" "configs/esm2_t36_3B_UR50D.yaml" "configs/esm2_t48_15B_UR50D.yaml" "configs/prot_t5_xl_bfd.yaml" "configs/prot_t5_xl_half_uniref50-enc.yaml")
    mkdir -p weights/
    case $ch in
        0 ) echo "Encoding sequences..."; 
            for task in "ss" "idp" "linker" "rdp"; do  # "ss" "idp" "linker" "rdp"
                python models/encoder.py --task $task
            done;;
        1 ) echo "Training secondary structure predictor..."
            rm -rf checkpoints
            python models/convert.py --config configs/ensemble_ss.yaml
            for emb_dir in "${emb_dirs[@]}"; do
                python models/train.py --path $emb_dir --task ss
            done
            mv checkpoints/ weights/ss/;;
        2 ) echo " Evaluating secondary structure predictor..."
            python models/ensemble.py --task ss;;
        3 ) echo "Training intrinsically disordered protein predictor..."
            python models/convert.py --config configs/ensemble_idp.yaml
            rm -rf checkpoints
            for emb_dir in "${emb_dirs[@]}"; do
                python models/train.py --path $emb_dir --task idp --test fldpnn
            done
            mv checkpoints/ weights/idp/;;
        4 ) echo "Training intrinsically disordered protein predictor (w/ ss)..."
            echo "Choose test dataset: fldpnn|casp|disorder723|mxd494|sl329|disprot"
            read test
            python models/convert.py --config configs/ensemble_idp.ss.yaml
            rm -rf checkpoints
            for emb_dir in "${emb_dirs[@]}"; do
                python models/train.py --path $emb_dir --task idp --test $test
            done
            mv checkpoints/ weights/idp.ss.$test/;;
        5 ) echo "Evaluating intrinsically disordered protein predictor..."
            python models/ensemble.py --task idp --shortcut False;;
        6 ) echo "Evaluating intrinsically disordered protein predictor (w/ ss)..."
            echo "Choose test dataset: fldpnn|casp|disorder723|mxd494|sl329|disprot|caid|caid2"
            read test
            python models/ensemble.py --task idp --test $test --shortcut True;;
        7 ) echo "Training disordered flexible linker predictor..."
            python models/convert.py --config configs/ensemble_linker.yaml
            rm -rf checkpoints
            for emb_dir in "${emb_dirs[@]}"; do
                python models/train.py --path $emb_dir --task linker
            done
            mv checkpoints/ weights/linker/;;
        8 ) echo "Training disordered flexible linker predictor (w/ ss)..."
            python models/convert.py --config configs/ensemble_linker.ss.yaml
            rm -rf checkpoints
            for emb_dir in "${emb_dirs[@]}"; do
                python models/train.py --path $emb_dir --task linker
            done
            mv checkpoints/ weights/linker.ss/;;
        9 ) echo "Evaluating disordered flexible linker predictor..."
            python models/ensemble.py --task linker --shortcut False;;
        10 ) echo "Evaluating disordered flexible linker predictor (w/ ss)..."
            python models/ensemble.py --task linker --shortcut True;;
        11 ) echo "Training disordered protein-binding predictor..."
            python models/convert.py --config configs/ensemble_prot.yaml
            rm -rf checkpoints
            for emb_dir in "${emb_dirs[@]}"; do
                python models/train.py --path $emb_dir --task prot
            done
            mv checkpoints/ weights/prot/;;
        12 ) echo "Training disordered protein-binding predictor (w/ ss)..."
            python models/convert.py --config configs/ensemble_prot.ss.yaml
            rm -rf checkpoints
            for emb_dir in "${emb_dirs[@]}"; do
                python models/train.py --path $emb_dir --task prot
            done
            mv checkpoints/ weights/prot.ss/;;
        13 ) echo "Evaluating disordered protein-binding predictor..."
            python models/ensemble.py --task prot --shortcut False;;
        14 ) echo "Evaluating disordered protein-binding predictor (w/ ss)..."
            python models/ensemble.py --task prot --shortcut True;;
        q ) break;;
    esac
done