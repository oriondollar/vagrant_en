import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

import selfies as sf
from vagrant.utils import *
from vagrant.preprocessing import StringPreprocessor

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

prop_key = {'alpha': 'GFN2_POLARIZABILITY_MOLECULAR',
            'totenergy': 'DFT_TOTAL_ENERGY',
            'logp': 'logp'}

def compute_mean_mad(prop_data):
    means = {}
    mads = {}
    for prop, vals in prop_data.items():
        vals = torch.tensor(vals)
        mean = torch.mean(vals)
        ma = torch.abs(vals - mean)
        mad = torch.mean(ma)
        means[prop] = mean
        mads[prop] = mad
    return means, mads

def preprocess_batch(batch, args):
    batch_size, n_nodes, _ = batch['positions'].size()
    atom_mask = batch['atom_mask'].view(batch_size * n_nodes, -1).to(args.device, args.dtype)
    atom_positions = batch['positions'].view(batch_size * n_nodes, -1).to(args.device, args.dtype)
    edge_mask = batch['edge_mask'].to(args.device, args.dtype)
    one_hot = batch['one_hot'].to(args.device, args.dtype)
    charges = batch['charges'].to(args.device, args.dtype)
    y_true = batch['src'].to(args.device)
    y0 = batch['tgt'].to(args.device)
    y_mask = batch['tgt_mask'].to(args.device)
    nodes = preprocess_nodes(one_hot, charges, args.charge_power, args.charge_scale, args.device)
    nodes = nodes.view(batch_size * n_nodes, -1)
    edges = get_adj_matrix(n_nodes, batch_size, args.device)
    props = []
    scaled_props = []
    for prop in args.properties:
        prop_vals = batch[prop].to(args.device, args.dtype)
        props.append(prop_vals)
        scaled_props.append(((prop_vals - args.means[prop]) / args.mads[prop]).view(-1,1).view(-1,1))
    edge_attr = None
    return nodes, atom_positions, edges, edge_attr, atom_mask,\
           edge_mask, n_nodes, y_true, y0, y_mask, props, scaled_props

def load_datasets(args, val_proportion=0.1, test_proportion=0.1):
    conf_file = f"qmugs{'_no_h' if args.remove_h else ''}_heavy_lt_{'{}'.format(args.max_heavy_atoms)}.npy"
    conf_path = os.path.join(args.data_dir, conf_file)
    data_file = 'summary_heavy_lt_{}.csv'.format(args.max_heavy_atoms)
    data_path = os.path.join(args.data_dir, data_file)

    ### Load data
    data = pd.read_csv(data_path)
    prop_data = {}
    for prop in args.properties:
        prop_data[prop] = data[prop].to_numpy()
    smiles = data.drop_duplicates(subset='smiles').smiles.to_list()
    selfies = [sf.encoder(smi) for smi in smiles]
    confs = np.load(conf_path)

    ### String preprocessing
    vocab, inv_vocab, vocab_weights = get_string_attrs(selfies, max_length=args.max_length+2)
    args.vocab = vocab
    args.inv_vocab = inv_vocab
    args.vocab_weights = vocab_weights.to(args.device)
    args.d_vocab = len(vocab)
    string_preproc = StringPreprocessor(args.vocab, args.max_length)
    src = string_preproc.preprocess(selfies)
    tgt = src[:,:-1]
    tgt_mask = make_std_mask(tgt, args.vocab['_'])
    string_data = (src, tgt, tgt_mask, smiles, selfies)

    ### Conformer preprocessing
    mol_id = confs[:,0].astype(int)
    mol_split_indices = np.nonzero(mol_id[:-1] - mol_id[1:])[0] + 1
    data_list = np.split(confs, mol_split_indices)
    print('# unique molecules:', len(data_list))

    perm_path = f"./data/qmugs/qmugs{'_no_h' if args.remove_h else ''}_{'{}'.format(args.max_heavy_atoms)}.npy"
    if not os.path.exists('./data/qmugs/qmugs_{}_permutation.npy'.format(args.max_heavy_atoms)):
        perm = np.random.permutation(len(data_list)).astype('int32')
        np.save(perm_path, perm)
    else:
        perm = np.load(perm_path)
    data_list = [data_list[i] for i in perm]

    n_mol = len(data_list)
    val_index = int(n_mol * val_proportion)
    test_index = val_index + int(n_mol * test_proportion)
    val_data, test_data, train_data = np.split(data_list, [val_index, test_index])

    train_conf_data = []
    for mol in train_data:
        conf_ids = mol[:,1].astype(int)
        conf_split_indices = np.nonzero(conf_ids[:-1] - conf_ids[1:])[0] + 1
        train_conf_data += np.split(mol, conf_split_indices)
    test_conf_data = []
    for mol in test_data:
        conf_ids = mol[:,1].astype(int)
        conf_split_indices = np.nonzero(conf_ids[:-1] - conf_ids[1:])[0] + 1
        test_conf_data += np.split(mol, conf_split_indices)
    val_conf_data = []
    for mol in val_data:
        conf_ids = mol[:,1].astype(int)
        conf_split_indices = np.nonzero(conf_ids[:-1] - conf_ids[1:])[0] + 1
        val_conf_data += np.split(mol, conf_split_indices)
    return train_conf_data, val_conf_data, test_conf_data, string_data, prop_data, args

class QMugsDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=False, drop_last=False, sampler=None):
        super().__init__(dataset, batch_size, shuffle=shuffle, collate_fn=collate_fn,
                         drop_last=drop_last, sampler=sampler)

class QMugsDataset(Dataset):
    def __init__(self, data_list, string_data, prop_data, transform=None):
        self.transform = transform
        self.data_list = data_list
        self.srcs, self.tgts, self.tgt_masks, self.smiles, self.selfies = string_data
        self.prop_data = prop_data

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data_list[idx]
        if self.transform:
            sample = self.transform(sample)
        mol_id = sample['mol_id'].item()
        conf_id = sample['conf_id'].item()
        sample['src'] = self.srcs[mol_id]
        sample['tgt'] = self.tgts[mol_id]
        sample['tgt_mask'] = self.tgt_masks[mol_id]
        for prop, vals in self.prop_data.items():
            sample[prop] = vals[conf_id]
        return sample

class QMugsTransform(object):
    def __init__(self, dataset_info, device):
        self.atomic_number_list = torch.Tensor(dataset_info['atomic_nb'])[None, :].int()
        self.max_charge = torch.max(self.atomic_number_list)
        self.device = device

    def __call__(self, data):
        n = data.shape[0]
        new_data = {}
        new_data['positions'] = torch.from_numpy(data[:, -3:])
        atom_types = torch.from_numpy(data[:, -4].astype(int)[:, None])
        one_hot = atom_types == self.atomic_number_list
        new_data['one_hot'] = one_hot
        new_data['mol_id'] = data[0,0].astype(int)
        new_data['conf_id'] = data[0,1].astype(int)
        new_data['charges'] = atom_types.squeeze(-1)
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
        return torch.tensor(np.array(props))
    elif props[0].dim() == 0:
        return torch.stack(props)
    else:
        return torch.nn.utils.rnn.pad_sequence(props, batch_first=True, padding_value=0)
