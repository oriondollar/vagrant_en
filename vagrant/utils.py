import re
import numpy as np
from itertools import permutations
from sklearn.preprocessing import MinMaxScaler
from qm9.rdkit_functions import build_xae_molecule

from rdkit import Chem
from rdkit.Chem.rdchem import BondType

import torch
from torch.autograd import Variable

################################################################################
############################# STRING PROCESSING ################################
################################################################################

def one_hot(x, categories):
    encoding = torch.tensor([int(x == cat) for cat in categories])
    return encoding

def decode_mols(encoded_tensors, inv_vocab):
    "Decodes tensor containing token ids into string"
    mols = []
    for i in range(encoded_tensors.shape[0]):
        encoded_tensor = encoded_tensors.cpu().numpy()[i,:]
        mol_string = ''
        for i in range(encoded_tensor.shape[0]):
            idx = encoded_tensor[i]
            if inv_vocab[idx] == '<end>':
                break
            else:
                mol_string += inv_vocab[idx]
        mol_string = mol_string.strip('_')
        mols.append(mol_string)
    return mols

def encode_smiles(tokenized_smile, max_length, vocab):
    "Converts tokenized SMILES string to list of token ids"
    tokenized_smile = ['<start>'] + tokenized_smile
    for i in range(max_length - len(tokenized_smile)):
        if i == 0:
            tokenized_smile.append('<end>')
        else:
            tokenized_smile.append('_')
    smile_vec = [vocab[c] for c in tokenized_smile]
    return smile_vec

def get_string_attrs(smiles, max_length, pad_weight=0.1):
    "Get vocab and weights for SMILES/SELFIES dataset"
    vocab_idx = 0
    vocab = {'<start>': vocab_idx}
    vocab_idx += 1
    tokenized_smiles = []
    for smi in smiles:
        tok_smi = tokenizer(smi)
        for tok in tok_smi:
            if tok not in vocab.keys():
                vocab[tok] = vocab_idx
                vocab_idx += 1
        tok_smi = ['<start>'] + tok_smi
        tok_smi.append('<end>')
        tokenized_smiles.append(tok_smi)
    vocab['<end>'] = vocab_idx
    vocab['_'] = vocab_idx + 1

    inv_vocab = {}
    for k, v in vocab.items():
        inv_vocab[v] = k

    vocab_weights = get_vocab_weights(tokenized_smiles, vocab, len(vocab.keys()),
                                      max_length)
    vocab_weights[-1] = pad_weight
    return vocab, inv_vocab, torch.tensor(vocab_weights).float()

def get_vocab_weights(train_smiles, vocab, num_char, max_length, freq_penalty=0.5):
    "Calculates token weights for a set of input data"
    char_dist = {}
    char_counts = np.zeros((num_char,))
    char_weights = np.zeros((num_char,))
    for k in vocab.keys():
        char_dist[k] = 0
    for smile in train_smiles:
        for i, char in enumerate(smile):
            char_dist[char] += 1
        for j in range(i, max_length):
            char_dist['_'] += 1
    for i, v in enumerate(char_dist.values()):
        char_counts[i] = v
    top = np.sum(np.log(char_counts))
    for i in range(char_counts.shape[0]):
        if char_counts[i] == 1:
            char_weights[i] = np.inf
        else:
            char_weights[i] = top / np.log(char_counts[i])
    min_weight = char_weights.min()
    for i, w in enumerate(char_weights):
        if w > 2*min_weight:
            char_weights[i] = 2*min_weight
    scaler = MinMaxScaler([freq_penalty,1.0])
    char_weights = scaler.fit_transform(char_weights.reshape(-1, 1))
    return char_weights[:,0]

def tokenizer(smile):
    "Tokenizes SMILES/SELFIES string"
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|_|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regezz = re.compile(pattern)
    tokens = [token for token in regezz.findall(smile)]
    assert smile == ''.join(tokens), ("{} could not be joined".format(smile))
    return tokens

################################################################################
######################### ENCODED STRING PROCESSING ############################
################################################################################

def make_std_mask(tgt, pad):
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
    return tgt_mask

def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

################################################################################
############################## QM9 BOND UTILS ##################################
################################################################################

bond_dict = {BondType.SINGLE: 1, BondType.DOUBLE: 2, BondType.TRIPLE: 3, BondType.AROMATIC: 4}

def pad_bonds(bonds):
    max_nodes = 0
    for mat in bonds:
        n_nodes = mat.shape[0]
        if n_nodes > max_nodes:
            max_nodes = n_nodes
    padded_mat = torch.zeros(len(bonds), max_nodes, max_nodes).long()
    for i, mat in enumerate(bonds):
        n_nodes = mat.shape[0]
        padded_mat[i,:n_nodes,:n_nodes] = mat
    return padded_mat

def get_bonds(positions, charges, smile, dataset_info):
    # Gather atom, adj and bond matrices
    x_dist, a_dist, e_dist = get_dist_adj_mats(positions, charges, dataset_info)
    x_rd, a_rd, e_rd = get_rdkit_adj_mats(smile, dataset_info)

    # Filter out hydrogens
    n_heavy_atoms = x_rd.shape[0]
    heavy_idxs = torch.where(x_dist > 0)[0]
    assert n_heavy_atoms == heavy_idxs.shape[0]

    # Create new tensors for reduced dist bond info
    x_heavy = torch.zeros(n_heavy_atoms).long()
    a_heavy = torch.zeros(n_heavy_atoms, n_heavy_atoms).long()
    e_heavy = torch.zeros(n_heavy_atoms, n_heavy_atoms).long()
    a_merge = a_dist.clone()
    e_merge = e_dist.clone()

    x_heavy = x_dist[heavy_idxs]
    for i, idx in enumerate(heavy_idxs):
        for j, jdx in enumerate(heavy_idxs):
            a_heavy[i,j] = a_dist[idx,jdx]
            e_heavy[i,j] = e_dist[idx,jdx]

    # Check for equivalency between rdkit and dist matrices
    # If not equivalent then attempt permuting atoms and inserting
    # aromatic bonds until equivalent or until best guess
    x_equiv = torch.equal(x_heavy, x_rd)
    a_equiv = torch.equal(a_heavy, a_rd)
    e_equiv = torch.equal(e_heavy, e_rd)
    if not x_equiv or not a_equiv or not e_equiv:
        x_heavy, a_heavy, e_heavy, a_min, e_min = swap_atoms_all_perms([x_heavy, a_heavy, e_heavy],
                                                                 [x_rd, a_rd, e_rd])

    for i, idx in enumerate(heavy_idxs):
        for j, jdx in enumerate(heavy_idxs):
            a_merge[idx,jdx] = a_heavy[i,j]
            e_merge[idx,jdx] = e_heavy[i,j]

    return a_merge.numpy(), e_merge.numpy()

def reorder_mat(mat, idxs):
    new_mat = torch.zeros(mat.shape).long()
    for i, idx in enumerate(idxs):
        for j, jdx in enumerate(idxs):
            new_mat[i,j] = mat[idx,jdx]
    for i in range(new_mat.shape[0]):
        for j in range(new_mat.shape[1]):
            if new_mat[i,j] > 0 and i < j:
                new_mat[j,i] = new_mat[i,j]
                new_mat[i,j] = 0
    return new_mat

def swap_atoms_all_perms(dist_mats, rdkit_mats):
    x_dist, a_dist, e_dist = dist_mats
    x_rd, a_rd, e_rd = rdkit_mats
    n_heavy_atoms = x_rd.shape[0]
    made_equiv = False
    perm_idxs = list(permutations(np.arange(n_heavy_atoms), n_heavy_atoms))
    a_min = 1e6
    e_min = 1e6
    for idxs in perm_idxs:
        idxs = list(idxs)
        x_rd_new = x_rd[idxs]
        if not torch.equal(x_rd_new, x_dist):
            continue
        a_rd_new = reorder_mat(a_rd, idxs)
        e_rd_new = reorder_mat(e_rd, idxs)
        e_dist_new = norm_aromatics(e_dist, e_rd_new)
        a_diff = np.abs(a_rd_new.numpy() - a_dist.numpy()).sum()
        e_diff = np.abs(e_rd_new.numpy() - e_dist_new.numpy()).sum()
        if a_diff <= a_min and e_diff <= e_min:
            a_min = a_diff
            e_min = e_diff
            x_rd_best = x_rd_new.clone()
            a_rd_best = a_rd_new.clone()
            e_rd_best = e_rd_new.clone()
        if torch.equal(x_rd_new, x_dist) and torch.equal(a_rd_new, a_dist) and torch.equal(e_rd_new, e_dist_new):
            return x_rd_new, a_rd_new, e_rd_new, a_min, e_min
    return x_rd_best, a_rd_best, e_rd_best, a_min, e_min

def norm_aromatics(e_dist, e_rd):
    e_dist_new = e_dist.clone()
    arom_locs = np.where(e_rd.numpy() == 4)
    for row, col in zip(arom_locs[0], arom_locs[1]):
        dist_type = e_dist[row,col]
        if dist_type == 1 or dist_type == 2:
            e_dist_new[row,col] = 4
    return e_dist_new

def get_dist_adj_mats(positions, charges, dataset_info):
    n_atoms = (charges > 0).sum().item()
    positions = positions[:n_atoms,:]
    charges = charges[:n_atoms]
    assert charges.shape[0] == (charges > 0).sum().item()
    atom_types = torch.tensor([dataset_info['charge2idx'][atom_num.item()] for atom_num in charges])
    x, a, e = build_xae_molecule(positions, atom_types, dataset_info)
    x = x.long()
    a = a.long()
    e = e.long()
    return x, a, e

def get_rdkit_adj_mats(smile, dataset_info):
    mol = Chem.MolFromSmiles(smile)
    n_atoms = mol.GetNumHeavyAtoms()
    x = torch.zeros(n_atoms).long()
    a = torch.zeros(n_atoms, n_atoms).long()
    e = torch.zeros(n_atoms, n_atoms).long()
    for j in range(n_atoms):
        atom = mol.GetAtomWithIdx(j)
        x[j] = dataset_info['charge2idx'][atom.GetAtomicNum()]
    for bond in mol.GetBonds():
        k = bond.GetBeginAtomIdx()
        l = bond.GetEndAtomIdx()
        bond_type = bond_dict[bond.GetBondType()]
        if l > k:
            a[l,k] = 1
            e[l,k] = bond_type
        else:
            a[k,l] = 1
            e[k,l] = bond_type
    return x, a, e
