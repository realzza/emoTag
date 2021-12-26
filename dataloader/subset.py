from torch._utils import _accumulate
from torch import randperm
from torch.utils.data import Dataset


class Subset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, sample_idx):
        if isinstance(sample_idx, int):
            return self.dataset[self.indices[sample_idx]]
        elif len(sample_idx) == 2:
            idx, tlen = sample_idx
            return self.dataset[(self.indices[idx], tlen)]
        else:
            raise AssertionError

    def __len__(self):
        return len(self.indices)


def random_split(dataset, valid_split):
    r"""
    Randomly split a dataset into non-overlapping new datasets of given valid_split percentage.

    Arguments:
        dataset (Dataset): Dataset to be split
        valid_split (sequence): percentage of validation data to be produced
    """
    assert 0 <= valid_split < 1
    valid_len = int(len(dataset) * valid_split)
    if valid_len == 0:
        return dataset, None
    train_len = len(dataset) - valid_len

    lengths = [train_len, valid_len]
    indices = randperm(sum(lengths)).tolist()
    return [Subset(dataset, indices[offset - length:offset]) for offset, length in zip(_accumulate(lengths), lengths)]
