import os
import torch
import logging
import numpy as np
import selfies as sf

from vagrant.utils import get_string_attrs

from torch.utils.data import DataLoader
from qm9.data.dataset import ProcessedDataset
from qm9.data.prepare import prepare_dataset

def initialize_datasets(args, datadir, dataset, calc_bonds=False, subset=None, splits=None,
                        force_download=False, reprocess=False, subtract_thermo=False,
                        seq_rep='selfies', max_length=125, remove_h=False):
    print('initializing...')
    # Set the number of points based upon the arguments
    num_pts = {'train': args.num_train,
               'test': args.num_test, 'valid': args.num_valid}

    # Download and process dataset. Returns datafiles.
    datafiles = prepare_dataset(
        datadir, dataset, calc_bonds, subset, splits, force_download=force_download, reprocess=reprocess)

    # Load downloaded/processed datasets
    datasets = {}
    for split, datafile in datafiles.items():
        with np.load(datafile) as f:
            split_data = {}
            for k, v in f.items():
                if k == 'smile':
                    split_data[k] = v
                else:
                    split_data[k] = torch.from_numpy(v)
            datasets[split] = split_data

    # Basic error checking: Check the training/test/validation splits have the same set of keys.
    keys = [list(data.keys()) for data in datasets.values()]
    assert all([key == keys[0] for key in keys]
               ), 'Datasets must have same set of keys!'

    # Remove hydrogens if requested
    if remove_h:
        for key, dataset in datasets.items():
            pos = dataset['positions']
            charges = dataset['charges']
            num_atoms = dataset['num_atoms']
            adjacency = dataset['adjacency']
            bonds = dataset['bonds']

            # Check that charges corresponds to real atoms
            assert torch.sum(num_atoms != torch.sum(charges > 0, dim=1)) == 0

            mask = dataset['charges'] > 1
            new_positions = torch.zeros_like(pos)
            new_charges = torch.zeros_like(charges)
            new_adjacency = torch.zeros_like(adjacency)
            new_bonds = torch.zeros_like(bonds)
            for i in range(new_positions.shape[0]):
                m = mask[i]
                keep_idxs = torch.where(m)[0]
                p = pos[i][m]
                p = p - torch.mean(p, dim=0)
                c = charges[i][m]
                n = torch.sum(m)
                new_positions[i, :n, :] = p
                new_charges[i,: n] = c
                for j, jdx in enumerate(keep_idxs):
                    for k, kdx in enumerate(keep_idxs):
                        new_adjacency[i,j,k] = adjacency[i,jdx,kdx]
                        new_bonds[i,j,k] = bonds[i,jdx,kdx]

            dataset['positions'] = new_positions
            dataset['charges'] = new_charges
            dataset['adjacency'] = new_adjacency
            dataset['bonds'] = new_bonds
            dataset['num_atoms'] = torch.sum(dataset['charges'] > 0, dim=1)

    # Get a list of all species across the entire dataset
    all_species = _get_species(datasets, ignore_check=False)

    # Get string data
    train_smiles = datasets['train']['smile']
    test_smiles = datasets['test']['smile']
    val_smiles = datasets['valid']['smile']
    if seq_rep == 'smiles':
        train_strings = train_smiles
        test_strings = test_smiles
        val_strings = val_smiles
    elif seq_rep == 'selfies':
        train_strings = np.empty(train_smiles.shape, dtype=np.object)
        test_strings = np.empty(test_smiles.shape, dtype=np.object)
        val_strings = np.empty(val_smiles.shape, dtype=np.object)
        for i, smi in enumerate(train_smiles):
            train_strings[i] = sf.encoder(smi)
        for i, smi in enumerate(test_smiles):
            test_strings[i] = sf.encoder(smi)
        for i, smi in enumerate(val_smiles):
            val_strings[i] = sf.encoder(smi)
    all_strings = np.concatenate([train_strings, test_strings, val_strings])
    vocab, inv_vocab, vocab_weights = get_string_attrs(all_strings, max_length=max_length+2)

    # Now initialize MolecularDataset based upon loaded data
    datasets = {split: ProcessedDataset(data, num_pts=num_pts.get(
        split, -1), included_species=all_species, subtract_thermo=subtract_thermo,
        vocab=vocab, inv_vocab=inv_vocab, vocab_weights=vocab_weights,
        seq_rep=seq_rep, max_length=max_length) for split, data in datasets.items()}

    # Check that all datasets have the same included species:
    assert(len(set(tuple(data.included_species.tolist()) for data in datasets.values())) ==
           1), 'All datasets must have same included_species! {}'.format({key: data.included_species for key, data in datasets.items()})

    # These parameters are necessary to initialize the network
    num_species = datasets['train'].num_species
    max_charge = datasets['train'].max_charge

    # Now, update the number of training/test/validation sets in args
    args.num_train = datasets['train'].num_pts
    args.num_valid = datasets['valid'].num_pts
    args.num_test = datasets['test'].num_pts

    return args, datasets, num_species, max_charge

def _get_species(datasets, ignore_check=False):
    # Get a list of all species in the dataset across all splits
    all_species = torch.cat([dataset['charges'].unique()
                             for dataset in datasets.values()]).unique(sorted=True)

    # Find the unique list of species in each dataset.
    split_species = {split: species['charges'].unique(
        sorted=True) for split, species in datasets.items()}

    # If zero charges (padded, non-existent atoms) are included, remove them
    if all_species[0] == 0:
        all_species = all_species[1:]

    # Remove zeros if zero-padded charges exst for each split
    split_species = {split: species[1:] if species[0] ==
                     0 else species for split, species in split_species.items()}

    # Now check that each split has at least one example of every atomic spcies from the entire dataset.
    if not all([split.tolist() == all_species.tolist() for split in split_species.values()]):
        # Allows one to override this check if they really want to. Not recommended as the answers become non-sensical.
        if ignore_check:
            logging.error(
                'The number of species is not the same in all datasets!')
        else:
            raise ValueError(
                'Not all datasets have the same number of species!')

    # Finally, return a list of all species
    return all_species
