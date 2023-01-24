import numpy as np

import torch
import torch.nn.functional as f

def mae(true, predicted):
    return np.abs(true - predicted).sum() / true.shape[0]

def kld_loss(mu, logvar, beta):
    kld = beta * -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    if torch.isnan(kld) or torch.isinf(kld):
        kld = torch.tensor(0.)
    return kld

def string_log_likelihood(y0, y_logits, vocab_weights):
    y0 = y0.long()[:,1:]
    y0 = y0.contiguous().view(-1)
    y_logits = y_logits.contiguous().view(-1, y_logits.size(2))
    bce = f.cross_entropy(y_logits, y0, reduction='mean', weight=vocab_weights)
    return bce

def vae_loss(y0, y_logits, mu, logvar, pred_props, true_props, beta, vocab_weights,
             device, anneal_mse=False):
    kld = kld_loss(mu, logvar, beta)
    bce = string_log_likelihood(y0, y_logits, vocab_weights)
    if pred_props is not None:
        mse = f.mse_loss(pred_props, true_props)
        if anneal_mse:
            mse *= beta
        if torch.isnan(mse) or torch.isinf(mse):
            mse = torch.tensor([0.]).to(device)
    else:
        mse = torch.tensor([0.]).to(device)
    return kld + bce + mse, kld, bce, mse
