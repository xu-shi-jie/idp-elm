# Accurate and Fast Prediction of Intrinsically Disordered Protein by Multiple Protein Language Models and Ensemble Learning

This is the official implementation of the paper ["Accurate and Fast Prediction of Intrinsically Disordered Protein by Multiple Protein Language Models and Ensemble Learning"](https://pubs.acs.org/doi/full/10.1021/acs.jcim.3c01202).

## Installation


```bash
conda create -n idpelm python=3.10
conda activate idpelm
pip install -r requirements.txt
```

Please also download all trained checkpoints from [here]() and decompress them into `weights/`.

## Usage

- It is strongly recommended to run the code on a machine with more than 70 GB RAM. Otherwise, you may encounter some memory issues.
- It is recommended to use `start_docker.sh` to create a docker container if you are using a Windows system.

To reproduce the results in the paper, please follow the steps below:

- Run `./train.sh`.
- After running, you have the following options:
  - [0] Before training models on datasets, you have to encode sequences into high-dimensional representations, to accelerate the training process.
  - [1] Train secondary structure predictor, which is used as a module in IDP-ELM.
  - [3, 7, 11] Train IDP-ELM and IDR function predictors.
  - Other options are explained in the script.

## Datasets
- IDP datasets
    - From [fIDPnn](https://www.nature.com/articles/s41467-021-24773-7), 
        - Training set: data/flDPnn/flDPnn_Training_Annotation.txt
        - Validation set: data/flDPnn/flDPnn_Validation_Annotation.txt
        - Test set: data/flDPnn/flDPnn_DissimiTest_Annotation.txt
    - From [CAID](https://www.nature.com/articles/s41592-021-01117-3)
        - Test set: data/CAID.txt
    - From [CAID2](https://onlinelibrary.wiley.com/doi/10.1002/prot.26582)
        - Test set: data/CAID2/disorder_pdb_2.txt
- DFL datasets 
    - From [DFLpred](https://academic.oup.com/bioinformatics/article/32/12/i341/2289031?login=true) 
        - Training set: data/DFL/DFL_benchmark_TR166/TR133.txt
        - Validation set: data/DFL/DFL_benchmark_TR166/DV33.txt
        - Test set: data/DFL/TE82.txt, data/DFL/TE64.txt (optional, see [here](http://bliulab.net/TransDFL/benchmark/))
- DP datasets
    - From [DeepDISOBind](https://pubmed.ncbi.nlm.nih.gov/34905768/)
        - Training set: data/DeepDISOBind/TrainingDataset.txt
        - Test set: data/DeepDISOBind/TestDataset.txt
- Secondary structure datasets
    - From [NetSurfP-3.0](https://academic.oup.com/nar/article/50/W1/W510/6596854)
        - Training set: data/NetSurfP-3.0/Train_HHblits.txt
        - Test set: 
            - CASP12: data/NetSurfP-3.0/CASP12_HHblits.txt
            - CB513: data/NetSurfP-3.0/CB513_HHblits.txt
            - TS115: data/NetSurfP-3.0/TS115_HHblits.txt

## Help

If you have any questions, please contact me at `shijie.xu@ees.hokudai.ac.jp`.
