import os
import sys
sys.path.append(os.getcwd())
import shutil
import tarfile
import argparse
import numpy as np
import pandas as pd
import multiprocessing
from tqdm import trange
from functools import partial

from rdkit import Chem
from vagrant.utils import chunks

DATA_DIR = '/gscratch/pfaendtner/orion/molecular_design/data/qmugs'

def build_chunk(members_chunk):
    chunk_id, members_chunk, row_idxs, mol_idxs = members_chunk
    print(chunk_id)
    tmp_sdf = "tmp/{}.sdf".format(chunk_id)

    dataset_conformers = []
    all_number_atoms = []
    for i in trange(len(members_chunk)):
        row_idx = row_idxs[i]
        mol_idx = mol_idxs[i]
        sdf_file = members_chunk[i]
        with open(tmp_sdf, 'w') as f:
            f.write(sdf_file.decode("utf-8"))
        mol = next(Chem.SDMolSupplier(tmp_sdf, removeHs=args.remove_h))
        n = mol.GetNumAtoms()
        all_number_atoms.append(n)
        coords = np.zeros((n, 4)).astype(float)
        conformer = mol.GetConformer()
        for j, atom in enumerate(mol.GetAtoms()):
            position = conformer.GetAtomPosition(j)
            charge = atom.GetAtomicNum()
            coords[j,:] = [charge, position.x, position.y, position.z]
        conf_id_arr = row_idx * np.ones((n, 1), dtype=float)
        mol_id_arr = mol_idx * np.ones((n, 1), dtype=float)
        id_coords = np.hstack((mol_id_arr, conf_id_arr, coords))
        dataset_conformers.append(id_coords)
    return [dataset_conformers, all_number_atoms]

def build(args):
    ### Read info
    os.makedirs('tmp', exist_ok=True)
    tar_path = os.path.join(DATA_DIR, args.struct_file)
    data_path = os.path.join(DATA_DIR, args.data_file)
    print('reading csv...')
    qmugs = pd.read_csv(data_path)
    print('reading tar...')
    with tarfile.open(tar_path) as tar:
        members = tar.getmembers()
        members_dict = {}
        print('extracting members...')
        for i in trange(len(members)):
            member = members[i]
            members_dict[member.name] = tar.extractfile(member).read()

    ### Write info
    write_dir = 'data/qmugs'
    try:
        data_desc = args.data_file.split('summary_')[1].split('.')[0]
    except IndexError:
        data_desc = None
    os.makedirs(write_dir, exist_ok=True)
    coords_file = f"qmugs{'_no_h' if args.remove_h else ''}{'_{}'.format(data_desc) if data_desc is not None else ''}"
    n_atoms_file = f"qmugs_n{'_no_h' if args.remove_h else ''}{'_{}'.format(data_desc) if data_desc is not None else ''}"

    ### Split data into chunks
    chembl_ids = list(set(qmugs.chembl_id.to_list()))
    chunk_size = len(chembl_ids) // args.n_threads
    chembl_chunks = list(chunks(chembl_ids, chunk_size))
    if len(chembl_chunks) > args.n_threads:
        chembl_chunks[-2] += chembl_chunks[-1]
    chembl_chunks = chembl_chunks[:-1]
    chembl_chunks = [[i, chunk] for i, chunk in enumerate(chembl_chunks)]
    chembl_chunk2id = {}
    for chunk in chembl_chunks:
        chunk_id, chunk = chunk
        for chembl_id in chunk:
            chembl_chunk2id[chembl_id] = chunk_id
    members_chunks = []
    for i in range(len(chembl_chunks)):
        members_chunks.append([])
    for chunk in chembl_chunks:
        chunk_id, chunk = chunk
        print('building data chunk {}...'.format(chunk_id))
        qmugs_chunk = qmugs[qmugs.chembl_id.isin(chunk)]
        chunk_idxs = qmugs_chunk.index
        members_data = []
        row_idxs = []
        mol_idxs = []
        for i in trange(qmugs_chunk.shape[0]):
            row = qmugs_chunk.iloc[i,:]
            row_idx = chunk_idxs[i]
            chembl_id = row.chembl_id
            conf_id = row.conf_id
            sdf_path = 'structures/{}/{}.sdf'.format(chembl_id, conf_id)
            members_data.append(members_dict[sdf_path])
            row_idxs.append(row_idx)
            mol_idxs.append(row.mol_id)
        members_chunks[chunk_id].append(chunk_id)
        members_chunks[chunk_id].append(members_data)
        members_chunks[chunk_id].append(row_idxs)
        members_chunks[chunk_id].append(mol_idxs)
    del members_dict
    print('# conformers:', qmugs.shape[0])
    print('# molecules:', len(chembl_ids))
    print('# threads available:', len(os.sched_getaffinity(0)))
    print('# threads:', args.n_threads)
    print('# molecules / chunk', chunk_size)

    ### Create parallel processes
    pool = multiprocessing.Pool(processes=args.n_threads)
    result = pool.map(build_chunk, members_chunks)
    dataset_conformers = []
    all_number_atoms = []
    for res in result:
        dataset_conformers += res[0]
        all_number_atoms += res[1]

    all_number_atoms = np.array(all_number_atoms)
    dataset = np.vstack(dataset_conformers)
    np.save(os.path.join(write_dir, coords_file), dataset)
    np.save(os.path.join(write_dir, n_atoms_file), all_number_atoms)
    shutil.rmtree('tmp')
    print('Dataset processed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--remove_h', default=False, action='store_true')
    parser.add_argument('--data_file', default='summary_heavy_lt_50.csv', type=str)
    parser.add_argument('--struct_file', default='structures.tar.gz', type=str)
    parser.add_argument('--n_threads', default=16, type=int)
    args = parser.parse_args()
    build(args)
