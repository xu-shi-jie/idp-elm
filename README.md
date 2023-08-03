# Accurate and Fast Prediction of Intrinsically Disordered Protein by Multiple Protein Language Models and Ensemble Learning

## Installatoin

`pip install -r requirements.txt`

## Usage

(It is recommended to use `start_docker.sh` to create a docker container if you are using Windows. )

For reproducing the results in the paper, please follow the steps below:

- Run `./train.sh`.
- After runing, you have following options:
  - [0] Before training models on datasets, you have to encode sequences into high-dimensional representations, for accelerating the training process.
  - [1] Train secondary structure predictor, which is used as a module in IDP-ELM.
  - [3, 7, 11] Train IDP-ELM and IDR function predictors.
  - Other options are explained as in the script.

## Help

If you have any questions, please contact me at `shijie.xu@ees.hokudai.ac.jp`.
