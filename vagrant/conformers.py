import torch
from tqdm import trange
from torch.nn.utils.rnn import pad_sequence
from vagrant.utils import get_bonds, pad_bonds
from morfeus.conformer import ConformerEnsemble


def conformer_to_data(conformer, included_species):
    positions = torch.tensor(conformer.coordinates).unsqueeze(0)
    charges = torch.tensor(conformer.elements).unsqueeze(0)
    one_hot = charges.unsqueeze(-1) == included_species.unsqueeze(0).unsqueeze(0)
    return positions, charges, one_hot

def smiles_to_coords(smiles, included_species, bond_types, dataset_info):
    mols = {"positions": [], "charges": [], "bonds": [], "smiles": []}
    succeeded = []
    for i in trange(len(smiles)):
        smi = smiles[i]
        try:
            ce = ConformerEnsemble.from_rdkit(smi, optimize="MMFF94")
            ce.prune_rmsd()
            ce.sort()
            conformer = ce.conformers[0]
            positions, charges, _ = conformer_to_data(conformer, included_species)
            positions = positions.squeeze(0)
            charges = charges.squeeze(0)
            adjacency, bonds = get_bonds(positions, charges, smi, dataset_info)
            mols['positions'].append(positions)
            mols['charges'].append(charges)
            mols['bonds'].append(torch.tensor(bonds))
            mols["smiles"].append(smi)
            succeeded.append(i)
        except:
            pass
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
