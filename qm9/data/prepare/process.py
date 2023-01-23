import os
import torch
import tarfile
import logging
from tqdm import trange
from torch.nn.utils.rnn import pad_sequence

from rdkit import Chem
from configs.datasets_config import get_dataset_info
from vagrant.utils import get_bonds, pad_bonds

charge_dict = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}

def split_dataset(data, split_idxs):
    split_data = {}
    for set, split in split_idxs.items():
        split_data[set] = {key: val[split] for key, val in data.items()}

    return split_data

def process_xyz_files(data, process_file_fn, calc_bonds, file_ext=None, file_idx_list=None, stack=True):
    logging.info('Processing data file: {}'.format(data))
    if tarfile.is_tarfile(data):
        tardata = tarfile.open(data, 'r')
        files = tardata.getmembers()

        readfile = lambda data_pt: tardata.extractfile(data_pt)

    elif os.is_dir(data):
        files = os.listdir(data)
        files = [os.path.join(data, file) for file in files]

        readfile = lambda data_pt: open(data_pt, 'r')

    else:
        raise ValueError('Can only read from directory or tarball archive!')

    # Use only files that end with specified extension.
    if file_ext is not None:
        files = [file for file in files if file.endswith(file_ext)]

    # Use only files that match desired filter.
    if file_idx_list is not None:
        files = [file for idx, file in enumerate(files) if idx in file_idx_list]

    # Now loop over files using readfile function defined above
    # Process each file accordingly using process_file_fn

    molecules = []

    for i in trange(len(files)):
        file = files[i]
        with readfile(file) as openfile:
            molecules.append(process_file_fn(openfile, calc_bonds))

    # Check that all molecules have the same set of items in their dictionary:
    props = molecules[0].keys()
    assert all(props == mol.keys() for mol in molecules), 'All molecules must have same set of properties/keys!'

    # Convert list-of-dicts to dict-of-lists
    molecules = {prop: [mol[prop] for mol in molecules] for prop in props}

    # If stacking is desireable, pad and then stack.
    if stack:
        for k, v in molecules.items():
            if k == 'adjacency' or k == 'bonds':
                molecules[k] = pad_bonds(v)
            elif k != 'smile':
                if v[0].dim() > 0:
                    molecules[k] = pad_sequence(v, batch_first=True)
                else:
                    molecules[k] = torch.stack(v)
        #molecules = {key: pad_sequence(val, batch_first=True) if val[0].dim() > 0 else torch.stack(val) for key, val in molecules.items()}

    return molecules

def process_xyz_gdb9(datafile, calc_bonds):
    xyz_lines = [line.decode('UTF-8') for line in datafile.readlines()]

    num_atoms = int(xyz_lines[0])
    mol_props = xyz_lines[1].split()
    mol_xyz = xyz_lines[2:num_atoms+2]
    mol_freq = xyz_lines[num_atoms+2]
    smile = xyz_lines[-2].split()[0]
    smile = Chem.MolToSmiles(Chem.MolFromSmiles(smile))

    atom_charges, atom_positions = [], []
    for line in mol_xyz:
        atom, posx, posy, posz, _ = line.replace('*^', 'e').split()
        atom_charges.append(charge_dict[atom])
        atom_positions.append([float(posx), float(posy), float(posz)])

    ### Derive bonds
    if calc_bonds:
        dataset_info = get_dataset_info('qm9', remove_h=False)
        adjacency, bonds = get_bonds(torch.tensor(atom_positions), torch.tensor(atom_charges), smile, dataset_info)
    else:
        adjacency = torch.empty(1,)
        bonds = torch.empty(1,)

    prop_strings = ['tag', 'index', 'A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']
    prop_strings = prop_strings[1:]
    mol_props = [int(mol_props[1])] + [float(x) for x in mol_props[2:]]
    mol_props = dict(zip(prop_strings, mol_props))
    mol_props['omega1'] = max(float(omega) for omega in mol_freq.split())

    molecule = {'num_atoms': num_atoms, 'charges': atom_charges, 'positions': atom_positions,
                'adjacency': adjacency, 'bonds': bonds, 'smile': smile}

    skip_keys = ['smile', 'adjacency', 'bonds']
    molecule.update(mol_props)
    for k, v in molecule.items():
        if k not in skip_keys:
            molecule[k] = torch.tensor(v)

    return molecule
