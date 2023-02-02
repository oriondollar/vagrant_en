import os
import re
import torch
from tqdm import trange
from torch.nn.utils.rnn import pad_sequence
from vagrant.utils import get_bonds, pad_bonds
from morfeus.conformer import ConformerEnsemble

def conformer_to_data(conformer, included_species, with_h):
    positions = torch.tensor(conformer.coordinates).unsqueeze(0)
    charges = torch.tensor(conformer.elements).unsqueeze(0)
    if not with_h:
        mask = (charges != 1).view(-1)
        charges = charges[:,mask]
        positions = positions[:,mask,:]
    one_hot = charges.unsqueeze(-1) == included_species.unsqueeze(0).unsqueeze(0)
    return positions, charges, one_hot

def smiles_to_coords(smiles, included_species, bond_types, dataset_info):
    mols = {"positions": [], "charges": [], "bonds": [], "smiles": []}
    succeeded = []
    for i in trange(len(smiles)):
        smi = smiles[i]
        ce = ConformerEnsemble.from_rdkit(smi, optimize="MMFF94")
        ce.prune_rmsd()
        ce.sort()
        conformer = ce.conformers[0]
        positions, charges, _ = conformer_to_data(conformer, included_species, dataset_info['with_h'])
        positions = positions.squeeze(0)
        charges = charges.squeeze(0)
        adjacency, bonds = get_bonds(positions, charges, smi, dataset_info)
        mols['positions'].append(positions)
        mols['charges'].append(charges)
        mols['bonds'].append(torch.tensor(bonds))
        mols["smiles"].append(smi)
        succeeded.append(i)
    smiles = mols['smiles']
    del mols['smiles']
    for k, v in mols.items():
        if k == 'bonds':
            mols[k] = pad_bonds(v)
        elif v[0].dim() > 0:
            mols[k] = pad_sequence(v, batch_first=True)
        else:
            mols[k] = torch.stack(v)
    positions = mols['positions']
    charges = mols['charges']
    bonds = mols['bonds']
    one_hot = charges.unsqueeze(-1) == included_species.unsqueeze(0).unsqueeze(0)
    one_hot_edges = bonds.unsqueeze(-1) == bond_types.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    return positions, charges, one_hot, one_hot_edges, smiles, succeeded

def get_conformers(smile, out_file):
    ce = ConformerEnsemble.from_rdkit(smile, optimize="MMFF94")
    ce.prune_rmsd()
    ce.sort()
    ce.write_xyz(out_file, ids=[1])


#############################################################
##################### DFT LOG PARSERS #######################
#############################################################

def parse_gaussian_log(fn):
    with open(fn) as f:
        lines = f.readlines()
        alpha_occ_idxs = []
        alpha = None
        gap = None
        for i, line in enumerate(lines):
            if 'Isotropic polarizability' in line:
                line = line.split()
                alpha = float(line[-2])
            if 'Alpha  occ. eigenvalues' in line:
                alpha_occ_idxs.append(i)
        if len(alpha_occ_idxs) > 0:
            last_alpha_occ_idx = alpha_occ_idxs[-1]
            last_alpha_vert_idx = last_alpha_occ_idx + 1
            homo = float(lines[last_alpha_occ_idx].split()[-1])
            lumo = float(lines[last_alpha_vert_idx].split()[-5])
            gap = (lumo - homo) * 627.5095 / 27.2116
    return alpha, gap

def check_normal_termination(contents):
    contents = ''.join(contents)
    matches = re.finditer('Normal termination', contents)
    if len(list(matches)) == 2:
        return True
    else:
        return False

def check_frequencies(contents):
    for line in contents:
        if 'Frequencies' in line:
            line = line.strip().split()
            freq1 = float(line[-3])
            freq2 = float(line[-2])
            freq3 = float(line[-1])
            if freq1 > 0 and freq2 > 0 and freq3 > 0:
                continue
            else:
                return False
    return True

def check_invalid(contents):
    contents = ''.join(contents)
    if 'The combination of multiplicity' and 'impossible' in contents:
        return True
    else:
        return False

def check_oscillations(contents, n_check=10):
    energies = []
    for line in contents:
        if 'SCF Done:  E(RB3LYP)' in line:
            line = line.strip().split()
            energies.append(float(line[4]))
    oscillation = (energies[-2] - energies[-1] > 0)
    for i in range(n_check):
        latest_oscillation = (energies[-(i+3)] - energies[-(i+2)] > 0)
        if latest_oscillation != oscillation:
            oscillation = latest_oscillation
        else:
            return False
    return True

def check_failure_mode(fn):
    contents = []
    with open(fn) as f:
        for line in f:
            contents.append(line)
    good_termination = check_normal_termination(contents)
    if good_termination:
        pos_frequencies = check_frequencies(contents)
        if pos_frequencies:
            failure_mode = 'none'
        else:
            failure_mode = 'failed_frequencies'
    elif not good_termination:
        if check_invalid(contents):
            failure_mode = 'invalid_molecule'
        elif check_oscillations(contents):
            failure_mode = 'oscillating_convergence'
        else:
            failure_mode = 'other'
    return failure_mode

