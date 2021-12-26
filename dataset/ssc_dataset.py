import os 
import numpy as np 
import math 
import random 
import soundfile
import librosa
import torch 
from torch.utils.data import Dataset 
from python_speech_features import ssc 


class SscDataset(Dataset):
    def __init__(self, wav_scp, utt2label=None, spk2int=None, ssc_kwargs={}, padding="wrap", cmn=True):
        """
        Params:
            wav_scp         - <utt> <wavpath>
            utt2label       - <utt> <lab>
            ssc_kwargs    - config of ssc features
            padding         - "wrap" or "constant"(zeros), for feature truncation, 
                              effective only if waveform length is less than truncated length.
            cmn             - whether perform mean normalization for feats.
        """    
        self.utt2wavpath = {x.split()[0]:x.split()[1] for x in open(wav_scp)}
        self.utt2label = self.init_label(utt2label, spk2int)
        self.utts = sorted(list(self.utt2wavpath.keys()))
        self.ssc_kwargs = ssc_kwargs
        self.padding = 'wrap' if padding == "wrap" else 'constant'
        self.cmn = cmn
        self.len = len(self.utts)


    def init_label(self, utt2label, spk2int=None):
        utt2lab = {x.split()[0]:x.split()[1] for x in open(utt2label)}
        if spk2int is None:
            spks = sorted(set(utt2lab.values()))
            spk2int = {spk:i for i, spk in enumerate(spks)}
        else:
            spk2int = {x.split()[0]:int(x.split()[1]) for x in open(spk2int)} 
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
        y = np.pad(y, (0, tlen - n), mode=padding)
        return y


    def extract_ssc(self, y, sr, cmn=False):
        feat = ssc(y, sr, winfunc=np.hamming, **self.ssc_kwargs)
        if cmn:
            feat -= np.mean(feat, axis=0, keepdims=True)
        return feat.astype('float32')


    def __getitem__(self, sample_idx):
        if isinstance(sample_idx, int):
            index, tlen = sample_idx, None
        elif len(sample_idx) == 2:
            index, tlen = sample_idx
        else:
            raise AssertionError
        
        utt = self.utts[index]
        # y, sr = soundfile.read(self.utt2wavpath[utt])
        y, sr = librosa.load(self.utt2wavpath[utt], sr=22050, mono=True)
        assert y.ndim == 1, 'Support mono channel only!'
        y = self.trun_wav(y, tlen, self.padding)
        feat = self.extract_ssc(y, sr, cmn=self.cmn)
        label = self.utt2label[utt]
        return utt, feat, label


    def __len__(self):
        return self.len

