import os
import numpy as np
import selfies as sf
from rdkit import Chem
from tqdm import trange

import torch
from torch import nn
import torch.nn.functional as F

from vagrant.gcl import EGCL, unsorted_segment_sum
from vagrant.utils import make_std_mask, decode_mols, is_valid, most_common
from vagrant.transformer import Trans, Deconv, LayerNorm, Embeddings,\
                                SublayerConnection, PositionwiseFeedForward,\
                                PositionalEncoding, clones

class Vagrant(nn.Module):
    def __init__(self, args, predict_property=False, ckpt_file=None):
        super(Vagrant, self).__init__()
        self.args = args
        self.predict_property = predict_property
        self.act_fn = nn.SiLU()
        self.device = args.device

        self.n_epochs = 0
        self.best_loss = np.inf
        self.state = {'name': self.args.name,
                      'epoch': self.n_epochs,
                      'model_state_dict': None,
                      'optimizer_state_dict': None,
                      'best_loss': self.best_loss,
                      'args': self.args}

        if ckpt_file is None:
            self.build()
        else:
            self.load(ckpt_file)

    def build(self):
        ### Encoder
        self.node_emb = nn.Linear(self.args.in_node_nf, self.args.d_model)
        for i in range(0, self.args.n_enc):
            self.add_module("gcl_%d" % i, EGCL(self.args.d_model, self.args.d_model, self.args.d_model,
                                               edges_in_d=self.args.in_edge_nf,
                                               nodes_att_dim=0,
                                               recurrent=True,
                                               attention=self.args.edge_attention))
        self.node_dec = nn.Sequential(
                nn.Linear(self.args.d_model, self.args.d_model),
                self.act_fn,
                nn.Linear(self.args.d_model, self.args.d_model))
        self.z_mean = nn.Linear(self.args.d_model, self.args.d_latent)
        self.z_var = nn.Linear(self.args.d_model, self.args.d_latent)

        ### Decoder
        self.seq_emb = nn.Sequential(
                Embeddings(self.args.d_model, self.args.d_vocab),
                PositionalEncoding(self.args.d_model, self.args.p_dropout, self.args.max_length+2),
                nn.Dropout(self.args.p_dropout))
        self.deconv = Deconv(self.args.d_model, self.args.d_latent)
        for i in range(0, self.args.n_dec):
            self.add_module("trans_%d" % i, Trans(self.args.n_heads, self.args.d_model,
                                                  self.args.d_ff, self.args.p_dropout))
        self.gen = nn.Linear(self.args.d_model, self.args.d_vocab)

        ### Property Predictor
        if self.predict_property:
            pred_layers = []
            for i in range(self.args.pred_depth):
                if i == 0:
                    pred_layers.append(nn.Linear(self.args.d_latent, self.args.pred_width))
                    pred_layers.append(self.act_fn)
                elif i == self.args.pred_depth - 1:
                    pred_layers.append(nn.Linear(self.args.pred_width, 1))
                else:
                    pred_layers.append(nn.Linear(self.args.pred_width, self.args.pred_width))
                    pred_layers.append(self.act_fn)
            self.pred = nn.Sequential(*pred_layers)
        self.to(self.args.device)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes):
        h = self.node_emb(h0)
        for i in range(0, self.args.n_enc):
            h = self._modules["gcl_%d" % i](h, edges, x, edge_mask, edge_attr=edge_attr,
                                            node_attr=None, n_nodes=n_nodes)
        h = self.node_dec(h)
        h = h * node_mask
        h = h.view(-1, n_nodes, self.args.d_model)
        if self.args.readout == 'sum':
            h_mol = torch.sum(h, dim=1)
        elif self.args.readout == 'mean':
            h_mol = torch.mean(h, dim=1)
        mu, logvar = self.z_mean(h_mol), self.z_var(h_mol)
        return mu, logvar

    def decode(self, y0, z, mask):
        y = self.seq_emb(y0)
        z = self.deconv(z)
        for i in range(0, self.args.n_dec):
            y = self._modules["trans_%d" % i](y, z, mask)
        y_logits = self.gen(y)
        return y_logits

    def forward(self, h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes, y0, y_mask):
        mu, logvar = self.encode(h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes)
        z = self.reparameterize(mu, logvar)
        y_logits = self.decode(y0, z, y_mask)
        if self.predict_property:
            pred_prop = self.pred(z)
        else:
            pred_prop = None
        return mu, logvar, y_logits, pred_prop

    ##########################################
    ############ VAGRANT METHODS #############
    ##########################################

    def save(self, prefix):
        os.makedirs(self.args.ckpt_dir, exist_ok=True)
        fn = '{}_{}.ckpt'.format(prefix, self.args.name)
        torch.save(self.state, os.path.join(self.args.ckpt_dir, fn))

    def load(self, load_fn, distributed=False):
        loaded_checkpoint = torch.load(load_fn, map_location=torch.device('cpu'))

        for k, v in loaded_checkpoint.items():
            try:
                self.state[k] = v
            except KeyError:
                self.state[k] = None

        model_state_dict = {}
        for k, v in self.state['model_state_dict'].items():
            if distributed:
                model_state_dict[''.join(k.split('module.'))] = v
            else:
                model_state_dict[k] = v

        self.n_epochs = self.state['epoch']
        self.best_loss = self.state['best_loss']
        self.args = self.state['args']
        self.build()
        self.load_state_dict(model_state_dict)

    def greedy_decode(self, z, return_ll=False):
        start_token = self.args.vocab['<start>']
        stop_token = self.args.vocab['<end>']
        pad_token = self.args.vocab['_']
        y_hat = torch.ones(z.shape[0],self.args.max_length+1).fill_(start_token).long()
        ll = torch.zeros(z.shape[0])
        freeze_ll = torch.ones(z.shape[0]).long()

        if self.args.cuda:
            y_hat = y_hat.cuda()

        self.eval()
        for i in range(self.args.max_length):
            y_mask = make_std_mask(y_hat, pad_token)
            if self.args.cuda:
                y_mask = y_mask.cuda()
            out = self.decode(y_hat, z, y_mask)
            prob = F.softmax(out[:,i,:], dim=-1)
            next_prob, next_word = torch.max(prob, dim=1)
            y_hat[:,i+1] = next_word
            next_prob = next_prob.detach().cpu()
            next_word = next_word.detach().cpu()
            freeze_ll[torch.where(next_word == stop_token)[0]] = 0
            ll[torch.where(freeze_ll == 1)[0]] += torch.log(next_prob)[torch.where(freeze_ll == 1)[0]]
            if freeze_ll.sum() == 0:
                break
        y_hat[:,i+1:] = pad_token
        y_hat = y_hat[:,1:]
        if return_ll:
            return y_hat, ll
        else:
            return y_hat

    def temp_decode(self, z, temp=0.5):
        start_token = self.args.vocab['<start>']
        stop_token = self.args.vocab['<end>']
        pad_token = self.args.vocab['_']
        y_hat = torch.ones(z.shape[0],self.args.max_length+1).fill_(start_token).long()
        freeze_ll = torch.ones(z.shape[0]).long()

        if self.args.cuda:
            y_hat = y_hat.cuda()

        self.eval()
        for i in range(self.args.max_length):
            y_mask = make_std_mask(y_hat, pad_token)
            if self.args.cuda:
                y_mask = y_mask.cuda()
            out = self.decode(y_hat, z, y_mask)[:,i,:] / temp
            prob = F.softmax(out, dim=-1)
            next_word = torch.distributions.Categorical(prob).sample()
            y_hat[:,i+1] = next_word
            next_word = next_word.detach().cpu()
            freeze_ll[torch.where(next_word == stop_token)[0]] = 0
            if freeze_ll.sum() == 0:
                break
        y_hat[:,i+1:] = pad_token
        y_hat = y_hat[:,1:]
        return y_hat

    def sample_direct(self, n_samples, from_z=False, z=None):
        batch_size = self.args.batch_size

        gen = []
        pred_props = torch.empty(0, 1)
        sampled_z = torch.empty(0,128)
        for i in trange(n_samples // batch_size):
            if not from_z:
                batch_z = torch.randn(batch_size, self.args.d_latent)
            else:
                batch_z = z[i*batch_size:(i+1)*batch_size]
            sampled_z = torch.cat([sampled_z, batch_z])
            if self.args.cuda:
                batch_z = batch_z.cuda()
            if self.args.predict_property:
                pred_prop = self.pred(batch_z)
                pred_prop = ((self.args.mad * pred_prop.detach().cpu()) + self.args.meann)
                pred_props = torch.cat([pred_props, pred_prop])
            else:
                pred_props = torch.cat([pred_props, torch.zeros(batch_size,1)])
            if self.args.decode_method == 'greedy':
                y_hat = self.greedy_decode(batch_z)
            elif self.args.decode_method == 'temp':
                y_hat = self.temp_decode(batch_z, temp=self.args.temp)
            batch_gen = decode_mols(y_hat, self.args.inv_vocab)
            gen += batch_gen
        if self.args.seq_rep == 'selfies':
            gen = [sf.decoder(selfie) for selfie in gen]
        gen = [Chem.MolToSmiles(Chem.MolFromSmiles(smi)) if is_valid(smi) else smi for smi in gen]
        return gen, pred_props, sampled_z

    def sample_robust(self, n_samples, n_perturbations, radius,
                      high_entropy_dims, low_entropy_dims,
                      from_z=False, z=None):
        gen = []
        pred_props = torch.empty(n_samples, 1)
        sampled_z = torch.empty(n_samples, 128)
        for i in trange(n_samples):
            if not from_z:
                center = torch.randn(128)
                center[low_entropy_dims] = 0
            else:
                center = z[i,:]
            structure = torch.zeros(n_perturbations, 128)
            structure[0,:] = center
            for j in range(1, n_perturbations):
                perturbation = center.clone()
                perturbation[high_entropy_dims] += torch.randn(high_entropy_dims.shape[0]) * radius
                structure[j,:] = perturbation
            structure = structure.cuda()
            if self.args.predict_property:
                perturbed_pred_props = self.pred(structure)
                perturbed_pred_props = ((self.args.mad * perturbed_pred_props.detach().cpu()) + self.args.meann)
            if self.args.decode_method == 'greedy':
                y_hat = self.greedy_decode(structure)
            elif self.args.decode_method == 'temp':
                y_hat = self.temp_decode(structure, temp=self.args.temp)
            batch_gen = decode_mols(y_hat, self.args.inv_vocab)
            if self.args.seq_rep == 'selfies':
                batch_gen = [sf.decoder(selfie) for selfie in batch_gen]
            batch_gen = [Chem.MolToSmiles(Chem.MolFromSmiles(smi)) if is_valid(smi) else smi for smi in batch_gen]
            sampled_smile = most_common(batch_gen)
            gen.append(sampled_smile)
            pred_idxs = []
            for j, smi in enumerate(batch_gen):
                if smi == sampled_smile:
                    pred_idxs.append(j)
            if self.args.predict_property:
                pred_props[i,:] = perturbed_pred_props[pred_idxs].mean()
            else:
                pred_props[i,:] = torch.zeros(1,)
            sampled_z[i,:] = torch.mean(structure[pred_idxs,:], dim=0)
        return gen, pred_props, sampled_z

    def __repr__(self):
        print_str = ''
        print_str += '\t# params: {}\n'.format(sum([x.nelement() for x in self.parameters()]))
        print_str += '\t# EGCL layers: {}\n'.format(self.args.n_enc)
        print_str += '\t# Trans layers: {}\n'.format(self.args.n_dec)
        print_str += '\tmodel dimensionality: {}\n'.format(self.args.d_model)
        print_str += '\tlatent dimensionality: {}\n'.format(self.args.d_latent)
        if self.args.predict_property:
            print_str += '\ttarget property: {}'.format(self.args.property)
        return print_str
