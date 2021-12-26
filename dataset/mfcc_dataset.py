import os 
import numpy as np 
import json
import math 
import random 
import soundfile
import h5py
import torch 
from torch.utils.data import Dataset 
from python_speech_features import mfcc 


class MfccDataset(Dataset):
    def __init__(self, wav_scp, utt2label=None, spk2int=None, mfcc_kwargs={}, padding="wrap", cmn=True, tdur=5):
        """
        Params:
            wav_scp         - <utt> <wavpath>
            utt2label       - <utt> <lab>
            mfcc_kwargs    - config of mfcc features
            padding         - "wrap" or "constant"(zeros), for feature truncation, 
                              effective only if waveform length is less than truncated length.
            cmn             - whether perform mean normalization for feats.
            tdur            - time duration of each utterance
        """    
        self.utt2wavpath = {x.split()[0]:x.split()[1] for x in open(wav_scp)}
        self.utt2label = self.init_label(utt2label, spk2int)
        self.utts = sorted(list(self.utt2wavpath.keys()))
        self.mfcc_kwargs = mfcc_kwargs
        self.padding = 'wrap' if padding == "wrap" else 'constant'
        self.cmn = cmn
        self.len = len(self.utts)
        self.tdur = tdur * 100 # 16k sample rate


    def init_label(self, utt2label, spk2int=None):
        utt2lab = {x.split()[0]:x.split()[1] for x in open(utt2label)}
        if spk2int is None:
            spks = sorted(set(utt2lab.values()))
            spk2int = {spk:i for i, spk in enumerate(spks)}
        else:
            with open(spk2int,'r') as f:
                spk2int = json.load(f)
#             spk2int = {x.split()[0]:int(x.split()[1]) for x in open(spk2int)} 
        utt2label = {utt:spk2int[spk] for utt, spk in utt2lab.items()}
        return utt2label
        
        
    def trun_wav(self, y, tlen, padding):
        """
        Truncation, zero padding or wrap padding for waveform.
        """
        # no need for truncation or padding
        if tlen is None:
            return y
        n = len(y)
        # needs truncation
        if n > tlen:
            offset = random.randint(0, n - tlen)
            y = y[offset:offset+tlen]
            return y
        # needs padding (zero/repeat padding)
        y = np.resize(y, (tlen, 13))
        return y


    def extract_feature(self, h5Dir):
        hf = h5py.File(h5Dir, 'r')
        mfccFeat = np.array(hf.get('mfcc'))
        hf.close()
        mfccFeat = self.trun_wav(mfccFeat, self.tdur, self.padding)
        return mfccFeat.astype('float32')


    def __getitem__(self, sample_idx):
        if isinstance(sample_idx, int):
            index, tlen = sample_idx, None
        elif len(sample_idx) == 2:
            index, tlen = sample_idx
        else:
            raise AssertionError
        
        utt = self.utts[index]
        feat = self.extract_feature(self.utt2wavpath[utt])
        label = self.utt2label[utt]
        return utt, feat, label
#         return utt, feat


    def __len__(self):
        return self.len

