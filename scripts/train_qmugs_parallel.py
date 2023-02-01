import os
import sys
sys.path.append(os.getcwd())
import json
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch import optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from qmugs.loss import vae_loss, mae
from qmugs import utils as qmugs_utils
from configs.datasets_config import get_dataset_info
from qmugs.utils import QMugsDataset, QMugsDataLoader,\
                        preprocess_batch, compute_mean_mad, prop_key

from vagrant.model import Vagrant
from vagrant.utils import KLAnnealer

def train(rank, datasets, args):
    print('in rank {}'.format(rank))
    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=args.n_gpus, rank=rank)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    args.device = torch.device('cuda:{}'.format(rank))
    print('rank {} initialized...'.format(rank))

    dataset_info = get_dataset_info('qmugs', remove_h=args.remove_h)
    transform = qmugs_utils.QMugsTransform(dataset_info, args.device)
    args.charge_scale = transform.max_charge
    args.in_node_nf = transform.atomic_number_list.shape[-1] * (args.charge_power + 1)
    args.in_edge_nf = 0
    raw_train, raw_string, raw_props = datasets
    dataset = QMugsDataset(raw_train, raw_string, raw_props, transform=transform)
    del raw_train, raw_string
    print('datasets created...')

    if len(args.properties) > 0:
        args.predict_property = True
    else:
        args.predict_property = False
    args.means, args.mads = compute_mean_mad(raw_props)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=args.n_gpus,
                                                              rank=rank, shuffle=True)
    loader = QMugsDataLoader(dataset, batch_size=args.batch_size, sampler=sampler)
    print('rank {} iter created...'.format(rank))

    #### Build model
    model = Vagrant(args, predict_property=args.predict_property)
    state = model.state
    model = model.to(args.device)
    args.vocab_weights = args.vocab_weights.to(args.device)
    model = DDP(model, device_ids=[rank])
    model.train()

    #### Set up training helpers
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)
    kl_annealer = KLAnnealer(args.beta_init, args.beta, args.kl_anneal_stop, args.kl_anneal_start)
    n_epochs = 0

    for epoch in range(args.n_epochs):
        args.beta = kl_annealer(epoch)
        total = len(loader.dataset) // args.batch_size // args.n_gpus
        for i, data in enumerate(tqdm(loader, total=total)):
            nodes, atom_positions, edges, edge_attr, atom_mask,\
            edge_mask, n_nodes, y_true, y0, y_mask, props, scaled_props = preprocess_batch(data, args)

            mu, logvar, y_logits, pred_props = model(h0=nodes, x=atom_positions, edges=edges,
                                                     edge_attr=edge_attr, node_mask=atom_mask,
                                                     edge_mask=edge_mask, n_nodes=n_nodes,
                                                     y0=y0, y_mask=y_mask)
            loss, kld, bce, mses = vae_loss(y_true, y_logits, mu, logvar,
                                            pred_props, scaled_props,
                                            args.beta, args.vocab_weights, args.device)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if args.predict_property:
                prop = args.properties[0]
                mae_score = mae(props[0].detach().cpu().numpy().reshape(-1,1),
                            ((args.mads[prop] * pred_props[0].detach().cpu()) + args.means[prop]).numpy())
            else:
                mae_score = 0

            log_file = open(args.log_fn, 'a')
            log_file.write('{},{},{},{},{},{},{}\n'.format(epoch,
                                                           i,
                                                           np.round(loss.item(), 5),
                                                           np.round(kld.item(), 5),
                                                           np.round(bce.item(), 5),
                                                           np.round(mses[0].item(), 5),
                                                           np.round(mae_score, 5)))
            log_file.close()

        lr_scheduler.step()
        if rank == 0:
            n_epochs += 1
            state['epoch'] = n_epochs
            state['model_state_dict'] = model.state_dict()
            state['optimizer_state_dict'] = optimizer.state_dict()

            if n_epochs % args.save_freq == 0:
                ckpt_fn = '{}_{}.ckpt'.format(n_epochs, args.name)
                torch.save(state, os.path.join(args.ckpt_dir, ckpt_fn))

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

    ### Distributed args
    parser.add_argument('--port', default='12355', type=str)
    parser.add_argument('--local_rank', default=-1)
    args = parser.parse_args()

    # Set device and dtype
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    args.dtype = torch.float32
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port
    args.seed = random.randint(0, 2**32-1)
    args.n_gpus = torch.cuda.device_count()
    print(args)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    args.ckpt_dir = os.path.join(args.ckpt_dir, args.name)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    args.log_fn = os.path.join(args.ckpt_dir, 'log.txt')
    log_file = open(args.log_fn, 'a')
    log_file.write('epoch,batch,loss,kld,bce,mse,mae\n')
    log_file.close()

    print('loading data...')
    args.properties = [prop_key[prop] for prop in args.properties]
    raw_train, raw_val, raw_test, raw_string, raw_props, args = qmugs_utils.load_datasets(args)
    raw_string = [arr.numpy() if isinstance(arr, torch.Tensor) else arr for arr in raw_string]
    datasets = [raw_train, raw_string, raw_props]
    print('data loaded...')

    mp.spawn(train, nprocs=args.n_gpus, args=(datasets, args,))
