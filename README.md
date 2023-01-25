# E(3) Equivariant Translation VAE for 3D Molecular Design
This repo contains the training and generation code for the model described in the attached submission to ICML 2023. We briefly describe how to use the repo below.

## Installation

We build the model architecture of Vagrant using PyTorch. Certain functions depend on [RDKit](https://www.rdkit.org/docs/Install.html) and we use [Morfeus](https://kjelljorner.github.io/morfeus/installation.html) for conformer generation. We also use standard python packages `numpy, pandas, scikit-learn, and scipy`. Otherwise the repo is entirely self-contained.

## Data

The training script automatically downloads the QM9 dataset and stores it in `qm9/temp/qm9` the first time it is run. You can force a redownload by adding the `--force_download` argument or reprocess the data by adding the `--reprocess_data` argument. We also include a pretrained checkpoint file in `checkpoints/vagrant`.

## Training

To train a model, run the following code in a terminal `python scripts/train.py`. The default hyperparameters match the hyperparameters used to train the version of Vagrant reported in the submission.

## Generation

You can generate molecules by calling `python scripts/gen.py --name vagrant --ckpt_epoch 100`. The `--name` and `--ckpt_epoch` arguments are rquired. Additional options can be passed to select a sampling method or additional hyperparameters. For instance, to use robust sampling and calculate the coherence of each sample, you would call `python scripts/gen.py --name vagrant --ckpt_epoch 100 --sample_method robust --calc_coherence`.

## Conformers

We also include code for generating structural conformers from the sampled molecules. These functions can be found in `vagrant/conformers.py`. Generating conformers for some samples can be slow. This may be the case when calculating coherence and can some times cause errors during generation. 
