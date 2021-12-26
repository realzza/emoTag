import torch
import math


def pad_zeros(vec, pad, dim):
    """
    padding vec with zeros
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    if vec.shape[dim] == pad:
        return vec
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size, dtype=vec.dtype)], dim=dim)


def pad_wrap(vec, pad, dim):
    """
    padding vec in repeating mode
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimemsion to pad
    """
    if vec.shape[dim] == pad:
        return vec
    N = math.ceil(pad / vec.shape[dim])
    vec = torch.cat([vec] * N, dim=dim)
    vec = torch.split(vec, pad, dim)[0]
    return vec


class PadCollate():
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch (xs, ys) of sequences
    """

    def __init__(self, dim, mode='zeros'):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        assert dim >= 1
        self.dim = dim - 1 # dim 0 is the batch_size, so remove it
        assert mode in ['zeros', 'wrap'], "Unsupported padding mode: %s " % mode
        self.mode = mode

    def pad_collate(self, batch):
        """
        args:
            batch - list of (feats, labels)

        reutrn:
            recos - record names (str), no padding
            xs    - a tensor of all feats in 'batch' after padding
            ys    - a tensor of all labels in batch
        """
        items = list(zip(*batch))
        assert len(items) == 2, "support only (xs, ys) in a batch"
        xs, ys = items
        assert xs[0].ndim == ys[0].ndim, "ndim of feats and labels must be the same"

        # find longest sequence
        max_len = max([x.shape[self.dim] for x in xs])
        # pad according to max_len
        if self.mode == 'zeros':
            xs = [pad_zeros(torch.as_tensor(x), pad=max_len, dim=self.dim) for x in xs]
            ys = [pad_zeros(torch.as_tensor(y), pad=max_len, dim=self.dim) for y in ys]
        elif self.mode == 'wrap':
            xs = [pad_wrap(torch.as_tensor(x), pad=max_len, dim=self.dim) for x in xs]
            ys = [pad_wrap(torch.as_tensor(y), pad=max_len, dim=self.dim) for y in ys]
        # stack all
        xs = torch.stack(xs, dim=0)
        ys = torch.stack(ys, dim=0)
        return xs, ys

    def __call__(self, batch):
        return self.pad_collate(batch)
