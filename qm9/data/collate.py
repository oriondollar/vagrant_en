import torch

def batch_stack(props):
    if isinstance(props[0], str):
        return props
    elif not torch.is_tensor(props[0]):
        return torch.tensor(props)
    elif props[0].dim() == 0:
        return torch.stack(props)
    else:
        return torch.nn.utils.rnn.pad_sequence(props, batch_first=True, padding_value=0)

def drop_zeros(props, to_keep):
    if not torch.is_tensor(props[0]):
        return props
    elif props[0].dim() == 0:
        return props
    else:
        return props[:, to_keep, ...]

def collate_fn(batch):
    batch = {prop: batch_stack([mol[prop] for mol in batch]) for prop in batch[0].keys()}

    to_keep = (batch['charges'].sum(0) > 0)

    string_props = ['src', 'tgt', 'tgt_mask', 'smile']
    edge_props = ['one_hot_edges']

    batch_ = {}
    for k, p in batch.items():
        if k in string_props:
            batch_[k] = p
        elif k in edge_props:
            keep_idxs = torch.where(to_keep)[0]
            batch_[k] = p[:,:keep_idxs.shape[0],:keep_idxs.shape[0],:]
        else:
            batch_[k] = drop_zeros(p, to_keep)
    batch = batch_
    #batch = {key: drop_zeros(prop, to_keep) for key, prop in batch.items()}

    atom_mask = batch['charges'] > 0
    batch['atom_mask'] = atom_mask

    #Obtain edges
    batch_size, n_nodes = atom_mask.size()
    edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)

    #mask diagonal
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask

    #edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)
    batch['edge_mask'] = edge_mask.view(batch_size * n_nodes * n_nodes, 1)

    return batch
