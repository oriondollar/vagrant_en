import torch
from torch.utils.data import Dataset

import logging
import numpy as np
import selfies as sf
from itertools import permutations
from vagrant.utils import make_std_mask
from vagrant.preprocessing import StringPreprocessor

from configs.datasets_config import get_dataset_info
from qm9.rdkit_functions import build_xae_molecule

from rdkit import Chem
from rdkit.Chem.rdchem import BondType

class ProcessedDataset(Dataset):
    def __init__(self, data, vocab, inv_vocab, vocab_weights, seq_rep, max_length=125, included_species=None, num_pts=-1, normalize=True, shuffle=True, subtract_thermo=True):

        self.data = data
        self.vocab = vocab
        self.inv_vocab = inv_vocab
        self.vocab_weights = vocab_weights

        if num_pts < 0:
            self.num_pts = len(data['charges'])
        else:
            if num_pts > len(data['charges']):
                logging.warning('Desired number of points ({}) is greater than the number of data points ({}) available in the dataset!'.format(num_pts, len(data['charges'])))
                self.num_pts = len(data['charges'])
            else:
                self.num_pts = num_pts

        # If included species is not specified
        if included_species is None:
            included_species = torch.unique(self.data['charges'], sorted=True)
            if included_species[0] == 0:
                included_species = included_species[1:]

        if subtract_thermo:
            thermo_targets = [key.split('_')[0] for key in data.keys() if key.endswith('_thermo')]
            if len(thermo_targets) == 0:
                logging.warning('No thermochemical targets included! Try reprocessing dataset with --force-download!')
            else:
                logging.info('Removing thermochemical energy from targets {}'.format(' '.join(thermo_targets)))
            for key in thermo_targets:
                data[key] -= data[key + '_thermo'].to(data[key].dtype)

        self.included_species = included_species
        bond_types = self.data['bonds'].unique()
        bond_types = bond_types[torch.where(bond_types > 0)[0]]
        self.bond_types = bond_types

        self.data['one_hot'] = self.data['charges'].unsqueeze(-1) == included_species.unsqueeze(0).unsqueeze(0)
        self.data['one_hot_edges'] = self.data['bonds'].unsqueeze(-1) == bond_types.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        smiles = self.data['smile']
        if seq_rep == 'smiles':
            strings = smiles
        elif seq_rep == 'selfies':
            strings = [sf.encoder(smi) for smi in smiles]
        string_preproc = StringPreprocessor(self.vocab, max_length)
        src = string_preproc.preprocess(strings)
        tgt = src[:,:-1]
        tgt_mask = make_std_mask(tgt, self.vocab['_'])
        self.data['src'] = src
        self.data['tgt'] = tgt
        self.data['tgt_mask'] = tgt_mask

        self.num_species = len(included_species)
        self.max_charge = max(included_species)

        self.parameters = {'num_species': self.num_species, 'max_charge': self.max_charge}

        # Get a dictionary of statistics for all properties that are one-dimensional tensors.
        self.calc_stats()

        if shuffle:
            self.perm = torch.randperm(len(data['charges']))[:self.num_pts]
        else:
            self.perm = None

    def calc_stats(self):
        self.stats = {key: (val.mean(), val.std()) for key, val in self.data.items() if type(val) is torch.Tensor and val.dim() == 1 and val.is_floating_point()}

    def convert_units(self, units_dict):
        for key in self.data.keys():
            if key in units_dict:
                self.data[key] *= units_dict[key]

        self.calc_stats()

    def __len__(self):
        return self.num_pts

    def __getitem__(self, idx):
        if self.perm is not None:
            idx = self.perm[idx]
        return {key: val[idx] for key, val in self.data.items()}
