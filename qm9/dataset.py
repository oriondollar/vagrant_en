import torch
from torch.utils.data import DataLoader
from qm9.data.utils import initialize_datasets
from qm9.args import init_argparse
from qm9.data.collate import collate_fn

def retrieve_dataloaders(batch_size, num_workers=1, seq_rep='selfies', max_length=125,
                         return_datasets=False, remove_h=False, calc_bonds=False, force_download=False,
                         reprocess_data=False):
    # Initialize dataloader
    args = init_argparse('qm9')
    args.force_download = force_download
    args.reprocess_data = reprocess_data
    args, datasets, num_species, charge_scale = initialize_datasets(args, args.datadir, 'qm9',
                                                                    calc_bonds=calc_bonds,
                                                                    subtract_thermo=args.subtract_thermo,
                                                                    force_download=args.force_download,
                                                                    reprocess=args.reprocess_data,
                                                                    seq_rep=seq_rep,
                                                                    max_length=max_length,
                                                                    remove_h=remove_h)

    qm9_to_eV = {'U0': 27.2114, 'U': 27.2114, 'G': 27.2114, 'H': 27.2114, 'zpve': 27211.4, 'gap': 27.2114, 'homo': 27.2114,
                 'lumo': 27.2114}

    for dataset in datasets.values():
        dataset.convert_units(qm9_to_eV)

    args.vocab = datasets['train'].vocab
    args.inv_vocab = datasets['train'].inv_vocab
    args.vocab_weights = datasets['train'].vocab_weights
    args.d_vocab = len(args.vocab)

    # Construct PyTorch dataloaders from datasets
    dataloaders = {split: DataLoader(dataset,
                                     batch_size=batch_size,
                                     shuffle=args.shuffle if (split == 'train') else False,
                                     num_workers=num_workers,
                                     collate_fn=collate_fn)
                         for split, dataset in datasets.items()}

    if return_datasets:
        return datasets, dataloaders, charge_scale, args

    return dataloaders, charge_scale, args
