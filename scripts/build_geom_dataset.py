import os
import sys
sys.path.path.append(os.getcwd())
import msgpack
import re
import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, SequentialSampler
import argparse
import selfies as sf

from vagrant.utils import tokenizer

def extract_conformers(args):
    drugs_file = args.data_file
    save_file = f"geom_drugs{'_no_h' if args.remove_h else ''}{'_{}'.format(args.conformations) if args.conformations is not None else ''}_{'{}energy'.format(args.energy_type)}"
    smiles_list_file = 'geom_drugs_smiles.txt'
    number_atoms_file = f"geom_drugs_n{'_no_h' if args.remove_h else ''}{'_{}'.format(args.conformations) if args.conformations is not None else ''}_{'{}energy'.format(args.energy_type)}"

    unpacker = msgpack.Unpacker(open(drugs_file, "rb"))

    all_smiles = []
    all_number_atoms = []
    dataset_conformers = []
    mol_id = 0
    conf_id = 0
    for i, drugs_1k in enumerate(unpacker):
        print(f"Unpacking file {i}...")
        for smiles, all_info in drugs_1k.items():
            selfie = sf.encoder(smiles)
            selfie_toks = tokenizer(selfie)
            if len(selfie_toks) <= 125:
                all_smiles.append(smiles)
                conformers = all_info['conformers']
                if args.conformations is not None:
                    all_energies = []
                    for conformer in conformers:
                        all_energies.append(conformer['{}energy'.format(args.energy_type)])
                    all_energies = np.array(all_energies)
                    argsort = np.argsort(all_energies)
                    lowest_energies = argsort[:args.conformations]
                    conformers = [conformers[id] for id in lowest_energies]
                for conformer in conformers:
                    coords = np.array(conformer['xyz']).astype(float)        # n x 4
                    if args.remove_h:
                        mask = coords[:, 0] != 1.0
                        coords = coords[mask]
                    n = coords.shape[0]
                    all_number_atoms.append(n)
                    mol_id_arr = mol_id * np.ones((n, 1), dtype=float)
                    conf_id_arr = conf_id * np.ones((n, 1), dtype=float)
                    tot_ener_arr = conformer['{}energy'.format(args.energy_type)] * np.ones((n, 1))
                    id_coords = np.hstack((conf_id_arr, mol_id_arr, tot_ener_arr, coords))

                    dataset_conformers.append(id_coords)
                    conf_id += 1
                mol_id += 1

    print("Total number of conformers saved", mol_id)
    all_number_atoms = np.array(all_number_atoms)
    dataset = np.vstack(dataset_conformers)

    print("Total number of atoms in the dataset", dataset.shape[0])
    print("Average number of atoms per molecule", dataset.shape[0] / mol_id)

    # Save conformations
    np.save(os.path.join(args.data_dir, save_file), dataset)
    # Save SMILES
    with open(os.path.join(args.data_dir, smiles_list_file), 'w') as f:
        for s in all_smiles:
            f.write('{}\n'.format(s))

    # Save number of atoms per conformation
    np.save(os.path.join(args.data_dir, number_atoms_file), all_number_atoms)
    print("Dataset processed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--remove_h", action='store_true', help="Remove hydrogens from the dataset.")
    parser.add_argument('--conformations', default=None, type=int)
    parser.add_argument('--energy_type', choices=['rel', 'total'], default='total')
    parser.add_argument("--data_dir", type=str, default='./data/geom/')
    parser.add_argument("--data_file", type=str, default='./data/geom/drugs_crude.msgpack')
    args = parser.parse_args()
    build(args)
