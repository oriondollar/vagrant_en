import os
import sys
sys.path.append(os.getcwd())
import json
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch import optim

from qmugs.loss import vae_loss, mae
from qmugs import utils as qmugs_utils
from qmugs.utils import QMugsDataLoader, preprocess_batch, compute_mean_mad, prop_key
from configs.datasets_config import get_dataset_info

from vagrant.model import Vagrant
from vagrant.utils import KLAnnealer

def run_epoch(model, args, optimizer, lr_scheduler,
              kl_annealer, loader, partition='train'):
    losses = []
    kld_losses = []
    bce_losses = []
    mse_losses = []
    for i in range(len(args.properties)):
        mse_losses.append([])
    mae_scores = []
    for i in range(len(args.properties)):
        mae_scores.append([])
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
        loss, kld, bce, mses = vae_loss(y_true, y_logits, mu, logvar,
                                       pred_props, scaled_props,
                                       args.beta, args.vocab_weights, args.device)
        if partition == 'train':
            loss.backward()
            optimizer.step()

        if args.predict_property:
            for j, prop in enumerate(args.properties):
                mae_score = mae(props[j].detach().cpu().numpy().reshape(-1,1),
                            ((args.mads[prop] * pred_props[j].detach().cpu()) + args.means[prop]).numpy())
                mae_scores[j].append(mae_score)
        else:
            for _ in range(len(args.properties)):
                mae_scores.append(0)

        losses.append(loss.item())
        kld_losses.append(kld.item())
        bce_losses.append(bce.item())
        for j in range(len(args.properties)):
            mse_losses[j].append(mses[j].item())

        if i % args.log_freq == 0:
            print(partition + ": Epoch %d \t Iteration %d \t loss %.4f" % (model.n_epochs, i,
                np.mean(losses[-10:])))

    lr_scheduler.step()
    avg_loss = np.mean(losses)
    avg_kld = np.mean(kld_losses)
    avg_bce = np.mean(bce_losses)
    avg_mses = []
    avg_maes = []
    for j in range(len(args.properties)):
        avg_mses.append(np.mean(mse_losses[j]))
        avg_maes.append(np.mean(mae_scores[j]))

    return avg_loss, avg_kld, avg_bce, avg_mses, avg_maes

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
    args.properties = [prop_key[prop] for prop in args.properties]
    dataset_info = get_dataset_info('qmugs', remove_h=args.remove_h)
    raw_train, raw_val, raw_test, raw_string, raw_props, args = qmugs_utils.load_datasets(args)
    transform = qmugs_utils.QMugsTransform(dataset_info, args.device)
    args.charge_scale = transform.max_charge
    args.in_node_nf = transform.atomic_number_list.shape[-1] * (args.charge_power + 1)
    args.in_edge_nf = 0
    dataloaders = {}
    for key, data_list in zip(['train', 'valid', 'test'], [raw_train, raw_val, raw_test]):
        dataset = qmugs_utils.QMugsDataset(data_list, raw_string, raw_props, transform=transform)
        shuffle = (key == 'train')
        dataloaders[key] = QMugsDataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle)
    del raw_train, raw_val, raw_test, raw_string

    if len(args.properties) > 0:
        args.predict_property = True
    else:
        args.predict_property = False
    args.means, args.mads = compute_mean_mad(raw_props)

    ### Build model
    model = Vagrant(args, predict_property=args.predict_property)
    print('vagrant built...')
    print(model)

    ### Set up training helpers
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)
    kl_annealer = KLAnnealer(args.beta_init, args.beta, args.kl_anneal_stop, args.kl_anneal_start)

    ### Set up logging file
    run_log = {'epochs': [], 'losses': [], 'kld_losses': [], 'bce_losses': [], 
               'best_val': 1e10, 'best_test': 1e10, 'best_epoch': 0}
    for i in range(len(args.properties)):
        run_log['mse_losses_{}'.format(i+1)] = []
        run_log['mae_scores_{}'.format(i+1)] = []

    print('training for {} epochs...'.format(args.n_epochs))
    for epoch in range(0, args.n_epochs):
        args.beta = kl_annealer(epoch)
        train_loss, _, _, _, _ = run_epoch(model, args, optimizer, lr_scheduler, kl_annealer,
                                           loader=dataloaders['train'], partition='train')
        
        if epoch % args.test_freq == 0:
            val_loss, _, _, _, _ = run_epoch(model, args, optimizer, lr_scheduler, kl_annealer,
                                             loader=dataloaders['valid'], partition='valid')
            test_loss, kld, bce, mses, mae_scores = run_epoch(model, args, optimizer, lr_scheduler, 
                                                            kl_annealer, loader=dataloaders['test'],
                                                            partition='test')
            run_log['epochs'].append(epoch)
            run_log['losses'].append(test_loss)
            run_log['kld_losses'].append(kld)
            run_log['bce_losses'].append(bce)
            for i, mse in enumerate(mses):
                run_log['mse_losses_{}'.format(i+1)].append(mse)
            for i, mae_score in enumerate(mae_scores):
                run_log['mae_scores_{}'.format(i+1)].append(mae_score)

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
    parser.add_argument('--name', default='vagrant_qmugs', type=str)
    parser.add_argument('--ckpt_dir', default='checkpoints', type=str)
    parser.add_argument('--save_freq', default=10, type=int)
    parser.add_argument('--test_freq', default=1, type=int)
    parser.add_argument('--log_freq', default=50, type=int)

    ### Model Hyperparameters
    parser.add_argument('--d_model', default=256, type=int) 
    parser.add_argument('--d_latent', default=128, type=int)
    parser.add_argument('--properties', nargs='+', default=[]) 

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
    parser.add_argument('--data_dir', default='./data/qmugs', type=str)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--charge_power', default=2, type=int)
    parser.add_argument('--max_length', default=125, type=int)
    parser.add_argument('--remove_h', default=False, action='store_true')
    parser.add_argument('--max_heavy_atoms', default=50, type=int)
    args = parser.parse_args()
    train(args)
