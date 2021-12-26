import torch
import random

class TruncateBatchSampler(object):
    ''' 
    for common BatchSampler, it generates: 
    Batch 0: [0, 1, 2, 3]
    Batch 1: [4, 5, 6, 7]
    ...

    For TrunBatchSampler, it generates:
    Batch 0: [(0,a), (1,a), (2,a), (3,a)]
    Batch 1: [(4,b), (5,b), (6,b), (7,b)]
    ...
    where a,b... are random values in trun_range.

    The generated sequences are fed into function `__getitem__` of class Dataset.
    It can be used for dynamic LSTM sequences or dynamic nframe-spectrogram CNN inputs.
    ''' 
    def __init__(self, sampler, batch_size=1, drop_last=False, trun_range=None, step=1):
        assert isinstance(trun_range, list) and len(trun_range) == 2
        self.sampler = sampler
        self.trun_lens = list(range(trun_range[0], trun_range[1]+1, step))
        self.step = step
        self.batch_size = batch_size
        self.drop_last = drop_last    

    def __iter__(self):
        batch = []
        n = random.choice(self.trun_lens)
        for idx in self.sampler:
            batch.append((idx, n))
            if len(batch) == self.batch_size:
                yield batch
                batch = []
                n = random.choice(self.trun_lens)
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
