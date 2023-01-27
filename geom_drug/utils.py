import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

import selfies as sf
from vagrant.utils import *
from vagrant.preprocessing import StringPreprocessor

def load_datasets(args, val_proportion=0.1, test_proportion=0.1):
    if args.conformations is not None:
        conf_suffix = '_{}'.format(args.conformations)
    else:
        conf_suffix = ''
    if args.remove_h:
        conf_file = os.path.join(args.data_dir, 'geom_drugs_no_h{}_totenergy.npy'.format(conf_suffix))
    else:
        conf_file = os.path.join(args.data_dir, 'geom_drugs{}_totenergy.npy'.format(conf_suffix))
    smiles_file = os.path.join(args.data_dir, 'geom_drugs_smiles.txt')

    ### String Preprocessing
    all_smiles = pd.read_csv(smiles_file, header=None).to_numpy()[:,0]
    all_selfies = [sf.encoder(smi) for smi in all_smiles]
    vocab, inv_vocab, vocab_weights = get_string_attrs(all_selfies, max_length=args.max_length+2)
    args.vocab = vocab
    args.inv_vocab = inv_vocab
    args.vocab_weights = vocab_weights.to(args.device)
    args.d_vocab = len(vocab)
    string_preproc = StringPreprocessor(args.vocab, args.max_length)
    src = string_preproc.preprocess(all_selfies)
    tgt = src[:,:-1]
    tgt_mask = make_std_mask(tgt, args.vocab['_'])
    string_data = (src, tgt, tgt_mask, all_smiles, all_selfies)

    ### Conformer Preprocessing
    all_data = np.load(conf_file)
    mol_id = all_data[:,1].astype(int)

    mol_split_indices = np.nonzero(mol_id[:-1] - mol_id[1:])[0] + 1
    data_list = np.split(all_data, mol_split_indices)
    del all_data

    perm = np.load('./data/geom/geom_permutation.npy')
    data_list = [data_list[i] for i in perm]

    num_mol = len(data_list)
    val_index = int(num_mol * val_proportion)
    test_index = val_index + int(num_mol * test_proportion)
    val_data, test_data, train_data = np.split(data_list, [val_index, test_index])

    train_conf_data = []
    for mol in train_data:
        conf_ids = mol[:,0].astype(int)
        conf_split_indices = np.nonzero(conf_ids[:-1] - conf_ids[1:])[0] + 1
        train_conf_data += np.split(mol, conf_split_indices)
    test_conf_data = []
    for mol in test_data:
        conf_ids = mol[:,0].astype(int)
        conf_split_indices = np.nonzero(conf_ids[:-1] - conf_ids[1:])[0] + 1
        test_conf_data += np.split(mol, conf_split_indices)
    val_conf_data = []
    for mol in val_data:
        conf_ids = mol[:,0].astype(int)
        conf_split_indices = np.nonzero(conf_ids[:-1] - conf_ids[1:])[0] + 1
        val_conf_data += np.split(mol, conf_split_indices)
    return train_conf_data, val_conf_data, test_conf_data, string_data, args

class GeomDrugsDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=False, drop_last=False, sampler=None):
        super().__init__(dataset, batch_size, shuffle=shuffle, collate_fn=collate_fn,
                         drop_last=drop_last, sampler=sampler)

class GeomDrugsDataset(Dataset):
    def __init__(self, data_list, string_data, transform=None):
        self.transform = transform
        self.data_list = data_list
        self.srcs, self.tgts, self.tgt_masks, self.smiles, self.selfies = string_data

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data_list[idx]
        if self.transform:
            sample = self.transform(sample)
        mol_id = sample['mol_id'].item()
        sample['src'] = torch.tensor(self.srcs[mol_id])
        sample['tgt'] = torch.tensor(self.tgts[mol_id])
        sample['tgt_mask'] = torch.tensor(self.tgt_masks[mol_id])
        return sample

class GeomDrugsTransform(object):
    def __init__(self, dataset_info, include_charges, device):
        self.atomic_number_list = torch.Tensor(dataset_info['atomic_nb'])[None, :]
        self.max_charge = torch.max(self.atomic_number_list)
        self.device = device
        self.include_charges = include_charges

    def __call__(self, data):
        n = data.shape[0]
        new_data = {}
        new_data['positions'] = torch.from_numpy(data[:, -3:])
        atom_types = torch.from_numpy(data[:, -4].astype(int)[:, None])
        one_hot = atom_types == self.atomic_number_list
        new_data['one_hot'] = one_hot
        new_data['energy'] = torch.tensor(data[0,2])
        new_data['mol_id'] = data[0,1].astype(int)
        if self.include_charges:
            new_data['charges'] = atom_types
        else:
            new_data['charges'] = torch.zeros(0, device=self.device)
        new_data['atom_mask'] = torch.ones(n, device=self.device)

        return new_data

def collate_fn(batch):
    batch = {prop: batch_stack([mol[prop] for mol in batch]) for prop in batch[0].keys()}

    atom_mask = batch['atom_mask']

    # Obtain edges
    batch_size, n_nodes = atom_mask.size()
    edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)

    # mask diagonal
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool,
                           device=edge_mask.device).unsqueeze(0)
    edge_mask *= diag_mask

    batch['edge_mask'] = edge_mask.view(batch_size * n_nodes * n_nodes, 1)

    return batch

def batch_stack(props):
    if isinstance(props[0], str):
        return props
    elif not torch.is_tensor(props[0]):
        return torch.tensor(props)
    elif props[0].dim() == 0:
        return torch.stack(props)
    else:
        return torch.nn.utils.rnn.pad_sequence(props, batch_first=True, padding_value=0)
