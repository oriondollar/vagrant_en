import re
import numpy as np
import selfies as sf
from scipy.stats import entropy
from itertools import permutations
from sklearn.preprocessing import MinMaxScaler
from qm9.rdkit_functions import build_xae_molecule

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import BondType

import torch
from torch.autograd import Variable

################################################################################
################################## METRICS #####################################
################################################################################

def calc_entropy(samples):
    es = []
    for i in range(samples.shape[1]):
        probs, bin_edges = np.histogram(samples[:,i], bins=1000, range=(-5., 5.), density=True)
        es.append(entropy(probs))
    return np.array(es)

def calc_coherence(gen, regen, regen_idxs, dist=False):
    coherence = []
    cur_idx = 0
    for i, smi in enumerate(gen):
        if i in regen_idxs:
            regen_smi = regen[cur_idx]
            sim = tanimoto_sim(smi, regen_smi)
            if dist:
                coherence.append(1-sim)
            else:
                coherence.append(sim)
            cur_idx += 1
        else:
            coherence.append(None)
    return coherence

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

def standardize_smiles(smiles):
    smiles = [sf.decoder(sf.encoder(smi)) for smi in smiles]
    return [Chem.MolToSmiles(Chem.MolFromSmiles(smi)) for smi in smiles]

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

def pad_bonds(bonds, max_nodes=None):
    if max_nodes is None:
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

################################################################################
############################## DATA UTILS ######################################
################################################################################

def compute_mean_mad(dataloaders, label_property):
    values = dataloaders['train'].dataset.data[label_property]
    meann = torch.mean(values)
    ma = torch.abs(values - meann)
    mad = torch.mean(ma)
    return meann, mad

def get_adj_matrix(n_nodes, batch_size, device, edges_dic={}):
    if n_nodes in edges_dic:
        edges_dic_b = edges_dic[n_nodes]
        if batch_size in edges_dic_b:
            return edges_dic_b[batch_size]
        else:
            # get edges for a single sample
            rows, cols = [], []
            for batch_idx in range(batch_size):
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        rows.append(i + batch_idx*n_nodes)
                        cols.append(j + batch_idx*n_nodes)

    else:
        edges_dic[n_nodes] = {}
        return get_adj_matrix(n_nodes, batch_size, device, edges_dic=edges_dic)

    edges = [torch.LongTensor(rows).to(device), torch.LongTensor(cols).to(device)]
    return edges

def preprocess_nodes(one_hot, charges, charge_power, charge_scale, device):
    charge_tensor = (charges.unsqueeze(-1) / charge_scale).pow(
        torch.arange(charge_power + 1., device=device, dtype=torch.float32))
    charge_tensor = charge_tensor.view(charges.shape + (1, charge_power + 1))
    atom_scalars = (one_hot.unsqueeze(-1) * charge_tensor).view(charges.shape[:2] + (-1,))
    return atom_scalars

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
    props = batch[args.property].to(args.device, args.dtype)
    scaled_props = ((props - args.meann) / args.mad).view(-1,1)
    if args.include_bonds:
        edge_attr = batch['one_hot_edges'].contiguous().view(batch_size*n_nodes*n_nodes,-1).to(args.device, args.dtype)
    else:
        edge_attr = None
    return nodes, atom_positions, edges, edge_attr, atom_mask,\
           edge_mask, n_nodes, y_true, y0, y_mask, props, scaled_props

def preprocess_batch_from_inputs(pos, one_hot, charges, one_hot_edges, args):
    batch_positions = pos
    batch_one_hot = one_hot.to(args.device, args.dtype)
    batch_charges = charges.to(args.device, args.dtype)
    batch_size, n_nodes, _ = batch_positions.size()
    atom_positions = batch_positions.view(batch_size * n_nodes, -1).to(args.device, args.dtype)
    atom_mask = batch_charges > 0
    edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0).to(args.device)
    edge_mask *= diag_mask
    edge_mask = edge_mask.view(batch_size * n_nodes * n_nodes, 1).to(args.device, args.dtype)
    atom_mask = atom_mask.view(batch_size * n_nodes, -1).to(args.device, args.dtype)
    nodes = preprocess_nodes(batch_one_hot, batch_charges, args.charge_power, args.charge_scale, args.device)
    nodes = nodes.view(batch_size * n_nodes, -1)
    edges = get_adj_matrix(n_nodes, batch_size, args.device)
    if args.include_bonds:
        edge_attr = one_hot_edges.contiguous().view(batch_size*n_nodes*n_nodes,-1).to(args.device, args.dtype)
    else:
        edge_attr = None
    return nodes, atom_positions, edges, edge_attr, atom_mask, edge_mask, n_nodes


################################################################################
############################## TRAINING UTILS ##################################
################################################################################

class KLAnnealer:
    """
    Scales KL weight (beta) linearly according to the number of epochs
    """
    def __init__(self, kl_low, kl_high, n_epochs, start_epoch):
        self.kl_low = kl_low
        self.kl_high = kl_high
        self.n_epochs = n_epochs
        self.start_epoch = start_epoch

        self.kl = (self.kl_high - self.kl_low) / (self.n_epochs - self.start_epoch)

    def __call__(self, epoch):
        k = (epoch - self.start_epoch) if epoch >= self.start_epoch else 0
        beta = self.kl_low + k * self.kl
        if beta > self.kl_high:
            beta = self.kl_high
        else:
            pass
        return beta

################################################################################
############################### RDKIT UTILS ####################################
################################################################################

def is_valid(smi):
    if smi is not None and smi != '':
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                return True
        except:
            return False
    return False

def tanimoto_sim(mol1, mol2, strict=False):
    mol1 = Chem.MolFromSmiles(mol1)
    mol2 = Chem.MolFromSmiles(mol2)
    if mol1 is None or mol2 is None:
        if strict:
            return 1
        else:
            return None
    else:
        fp1 = AllChem.GetMorganFingerprint(mol1,2)
        fp2 = AllChem.GetMorganFingerprint(mol2,2)
        tan_sim = DataStructs.TanimotoSimilarity(fp1, fp2)
        return tan_sim

def most_common(lst):
    return max(set(lst), key=lst.count)
