import os
import sys
sys.path.append(os.getcwd())
import torch
import argparse

from qm9 import dataset

def train(args):
    # Set device and dtype
    args.use_gpu = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.use_gpu else "cpu")
    args.dtype = torch.float32

    os.makedirs(args.ckpt_dir, exist_ok=True)
    args.ckpt_dir = os.path.join(args.ckpt_dir, args.name)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    dataloaders, charge_scale, data_args = dataset.retrieve_dataloaders(args.batch_size,
                                                                        args.num_workers,
                                                                        args.seq_rep,
                                                                        args.max_length,
                                                                        remove_h=args.remove_h,
                                                                        calc_bonds=args.calc_bonds,
                                                                        force_download=args.force_download)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ### I/O Parameters
    parser.add_argument('--name', default='vagrant', type=str)
    parser.add_argument('--ckpt_dir', default='checkpoints', type=str)
    parser.add_argument('--save_freq', default=10, type=int)

    ### Model Hyperparameters
    parser.add_argument('--d_model', default=256, type=int) #nf (egnn - encoder only)
    parser.add_argument('--d_latent', default=128, type=int)
    parser.add_argument('--property', default='alpha', 
                        choices=['alpha', 'gap', 'homo', 'lumo',
                                 'mu', 'Cv', 'G', 'H', 'r2', 'U',
                                 'U0', 'zpve']) #property_name (Mollander-Z)
    parser.add_argument('--predict_property', default=False, action='store_true')

    ### Encoder Hyperparameters
    parser.add_argument('--n_enc', default=4, type=int) #n_layers
    parser.add_argument('--attention', default=1, type=int)
    parser.add_argument('--pred_depth', default=3, type=int) #pp_depth
    parser.add_argument('--pred_width', default=256, type=int) #pp_width
    parser.add_argument('--node_attr', default=0, type=int)
    parser.add_argument('--readout', default='sum', choices=['sum', 'mean'], type=str)

    ### Decoder Hyperparameters
    parser.add_argument('--n_dec', default=4, type=int) #N
    parser.add_argument('--d_ff', default=256, type=int)
    parser.add_argument('--n_heads', default=4, type=int)
    parser.add_argument('--p_dropout', default=0.0, type=float)
    parser.add_argument('--max_length', default=125, type=int)

    ### Training Hyperparameters
    parser.add_argument('--n_epochs', default=100, type=int) #epochs (egnn)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--beta', default=0.5, type=float)
    parser.add_argument('--beta_init', default=1e-8, type=float)
    parser.add_argument('--kl_anneal_start', default=0, type=int)
    parser.add_argument('--kl_anneal_stop', default=100, type=int)
    parser.add_argument('--weight_decay', default=1e-16, type=float)

    ### Data Hyperparameters
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--charge_power', default=2, type=int)
    parser.add_argument('--seq_rep', default='selfies', choices=['selfies', 'smiles'], type=str)
    parser.add_argument('--force_download', default=False, action='store_true')
    parser.add_argument('--calc_bonds', default=False, action='store_true')
    parser.add_argument('--remove_h', default=False, action='store_true')
    args = parser.parse_args()
    train(args)
