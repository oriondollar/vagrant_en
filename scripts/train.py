import os
import sys
sys.path.append(os.getcwd())
import json
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch import optim

from qm9 import dataset
from vagrant.model import Vagrant
from vagrant.loss import vae_loss, mae
from vagrant.utils import compute_mean_mad, preprocess_nodes, get_adj_matrix,\
                          preprocess_batch, KLAnnealer

def run_epoch(model, args, optimizer, lr_scheduler,
              kl_annealer, loader, partition='train'):
    epoch_log = {'loss': 0, 'counter': 0}
    kld_losses = []
    bce_losses = []
    mse_losses = []
    mae_scores = []
    total = len(loader.dataset) // args.batch_size
    for i, data in enumerate(tqdm(loader, total=total)):
        if partition == 'train':
            model.train()
            optimizer.zero_grad()
        else:
            model.eval()

        nodes, atom_positions, edges, edge_attr, atom_mask,\
        edge_mask, n_nodes, y_true, y0, y_mask, props, scaled_props = preprocess_batch(data, args)

        # Forward pass
        mu, logvar, y_logits, pred_props = model(h0=nodes, x=atom_positions, edges=edges,
                                                 edge_attr=edge_attr, node_mask=atom_mask,
                                                 edge_mask=edge_mask, n_nodes=n_nodes,
                                                 y0=y0, y_mask=y_mask)
        loss, kld, bce, mse = vae_loss(y_true, y_logits, mu, logvar,
                                       pred_props, scaled_props.view(-1,1),
                                       args.beta, args.vocab_weights, args.device)
        if partition == 'train':
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        if args.predict_property:
            mae_score = mae(props.detach().cpu().numpy().reshape(-1,1),
                            ((args.mad * pred_props.detach().cpu()) + args.meann).numpy())
        else:
            mae_score = 0

        epoch_log['loss'] += loss.item() * args.batch_size
        epoch_log['counter'] += args.batch_size
        kld_losses.append(kld.item())
        bce_losses.append(bce.item())
        mse_losses.append(mse.item())
        mae_scores.append(mae_score)
        avg_loss = epoch_log['loss'] / epoch_log['counter']

        if i % args.log_freq == 0:
            print(partition + ": Epoch %d \t Iteration %d \t loss %.4f" % (model.n_epochs, i,
                  epoch_log['loss'][-1] / args.batch_size))

    return avg_loss, np.mean(kld_losses), np.mean(bce_losses), np.mean(mse_losses), np.mean(mae_scores)


def train(args):
    # Set device and dtype
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    args.dtype = torch.float32

    os.makedirs(args.ckpt_dir, exist_ok=True)
    args.ckpt_dir = os.path.join(args.ckpt_dir, args.name)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    args.log_fn = os.path.join(args.ckpt_dir, 'log.json')

    ### Load data and set model arguments
    dataloaders, charge_scale, data_args = dataset.retrieve_dataloaders(args.batch_size,
                                                                        args.num_workers,
                                                                        args.seq_rep,
                                                                        args.max_length,
                                                                        remove_h=args.remove_h,
                                                                        calc_bonds=args.calc_bonds,
                                                                        force_download=args.force_download,
                                                                        reprocess_data=args.reprocess_data)
    print('data loaded...')
    args.vocab = data_args.vocab
    args.inv_vocab = data_args.inv_vocab
    args.vocab_weights = data_args.vocab_weights.to(args.device)
    args.d_vocab = data_args.d_vocab
    args.charge_scale = charge_scale
    meann, mad = compute_mean_mad(dataloaders, args.property)
    args.meann = meann
    args.mad = mad

    if args.remove_h:
        args.in_node_nf = 12
    else:
        args.in_node_nf = 15
    if args.include_bonds:
        args.in_edge_nf = 4
    else:
        args.in_edge_nf = 0

    ### Build model
    model = Vagrant(args, predict_property=args.predict_property, ckpt_file='checkpoints/vagrant/3000_vagrant.ckpt')
    print('vagrant built...')
    print(model)

    ### Set up training helpers
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)
    kl_annealer = KLAnnealer(args.beta_init, args.beta, args.kl_anneal_stop, args.kl_anneal_start)

    ### Set up logging file
    run_log = {'epochs': [], 'losess': [], 'kld_losses': [], 'bce_losses': [], 'mse_losses': [],
               'mae_scores': [], 'best_val': 1e10, 'best_test': 1e10, 'best_epoch': 0}


    print('training for {} epochs...'.format(args.n_epochs))
    for epoch in range(0, args.n_epochs):
        args.beta = kl_annealer(epoch)
        train_loss, _, _, _, _ = run_epoch(model, args, optimizer, lr_scheduler, kl_annealer,
                                           loader=dataloaders['train'], partition='train')
        
        if epoch % args.test_freq == 0:
            val_loss, _, _, _, _ = run_epoch(model, args, optimizer, lr_scheduler, kl_annealer,
                                             loader=dataloaders['valid'], partition='valid')
            test_loss, kld, bce, mse, mae_score = run_epoch(model, args, optimizer, lr_scheduler, 
                                                            kl_annealer, loader=dataloaders['test'],
                                                            partition='test')
            run_log['epochs'].append(epoch)
            run_log['losess'].append(test_loss)
            run_log['kld_losses'].append(kld)
            run_log['bce_losses'].append(bce)
            run_log['mse_losses'].append(mse)
            run_log['mae_scores'].append(mae_score)

            if val_loss < run_log['best_val']:
                run_log['best_val'] = val_loss
                run_log['best_test'] = test_loss
                run_log['best_epoch'] = epoch
                model.state['best_loss'] = val_loss
            print("Val loss: %.4f \t test loss: %.4f \t epoch %d" % (val_loss, test_loss, epoch))
            print("Best: val loss: %.4f \t test loss: %.4f \t epoch %d" % (run_log['best_val'],
                  run_log['best_test'], run_log['best_epoch']))

        model.n_epochs += 1
        model.state['epoch'] = model.n_epochs
        model.state['model_state_dict'] = model.state_dict()
        model.state['optimizer_state_dict'] = optimizer.state_dict()

        if model.n_epochs % args.save_freq == 0:
            epoch_str = str(model.n_epochs)
            while len(epoch_str) < 3:
                epoch_str = '0' + epoch_str
            model.save(epoch_str)

        json_object = json.dumps(run_log, indent=4)
        with open(args.log_fn, 'w') as fo:
            fo.write(json_object)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ### I/O Parameters
    parser.add_argument('--name', default='vagrant', type=str)
    parser.add_argument('--ckpt_dir', default='checkpoints', type=str)
    parser.add_argument('--save_freq', default=10, type=int)
    parser.add_argument('--test_freq', default=1, type=int)
    parser.add_argument('--log_freq', default=50, type=int)

    ### Model Hyperparameters
    parser.add_argument('--d_model', default=256, type=int) 
    parser.add_argument('--d_latent', default=128, type=int)
    parser.add_argument('--property', default='alpha', 
                        choices=['alpha', 'gap', 'homo', 'lumo',
                                 'mu', 'Cv', 'G', 'H', 'r2', 'U',
                                 'U0', 'zpve']) 
    parser.add_argument('--predict_property', default=False, action='store_true')

    ### Encoder Hyperparameters
    parser.add_argument('--n_enc', default=4, type=int) 
    parser.add_argument('--pred_depth', default=3, type=int) 
    parser.add_argument('--pred_width', default=256, type=int) 
    parser.add_argument('--readout', default='sum', choices=['sum', 'mean'], type=str)
    parser.add_argument('--edge_attention', default=True, type=bool)

    ### Decoder Hyperparameters
    parser.add_argument('--n_dec', default=4, type=int)
    parser.add_argument('--d_ff', default=256, type=int)
    parser.add_argument('--n_heads', default=4, type=int)
    parser.add_argument('--p_dropout', default=0.0, type=float)

    ### Training Hyperparameters
    parser.add_argument('--n_epochs', default=100, type=int) 
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
    parser.add_argument('--max_length', default=125, type=int)
    parser.add_argument('--force_download', default=False, action='store_true')
    parser.add_argument('--reprocess_data', default=False, action='store_true')
    parser.add_argument('--calc_bonds', default=False, action='store_true')
    parser.add_argument('--remove_h', default=False, action='store_true')
    parser.add_argument('--include_bonds', default=False, action='store_true')
    args = parser.parse_args()
    train(args)
