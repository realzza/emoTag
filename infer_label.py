import os
import json
import torch
import librosa
import argparse
import numpy as np
from tqdm import tqdm
from module.model import Gvector
from scipy.special import softmax
from python_speech_features import logfbank, mfcc
from numpy import argmax

def parse_args():
    desc="parse model info"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--vad_file', type=str, default="/Netdata/2018/wangwq/workspace/DeepSpeaker/egs/DiaPIPE/egs/voxconverse/affinity/sp_lstm_rotate/exp/pyr/subsegments_1.28_0.64/results/pred_rttm_100")
    parser.add_argument('--model_dir', type=str, default='exp/emo_resnet18_cosine_T20/chkpt/chkpt_best.pth')
    parser.add_argument('--data_dir', type=str, default="/NASdata/Teamwork-clip-audio/", help="data directory to be labelled")
    parser.add_argument('--output_dir', type=str, default="./labels/")
    return parser.parse_args()

# config (please keep the same settings with `conf/logfbank_train-emo.json`)
mdl_kwargs = {
    "channels": 16, 
    "block": "BasicBlock", 
    "num_blocks": [2,2,2,2], 
    "embd_dim": 1024, 
    "drop": 0.5, 
    "n_class": 5
}

fbank_kwargs = {
    "winlen": 0.025, 
    "winstep": 0.01, 
    "nfilt": 256, 
    "nfft": 1024, 
    "lowfreq": 0, 
    "highfreq": None, 
    "preemph": 0.97    
}


class SVExtractor():
    def __init__(self, mdl_kwargs, fbank_kwargs, resume, device):
        self.model = self.load_model(mdl_kwargs, resume)
        self.model.eval()
        self.device = device
        self.model = self.model.to(self.device)
        self.fbank_kwargs = fbank_kwargs

    def load_model(self, mdl_kwargs, resume):
        model = Gvector(**mdl_kwargs)
        state_dict = torch.load(resume,map_location=torch.device('cpu'))
        if 'model' in state_dict.keys():
            state_dict = state_dict['model']
        model.load_state_dict(state_dict)
        return model

    def extract_fbank(self, y, sr, cmn=True):
        feat = logfbank(y, sr, **self.fbank_kwargs)
        if cmn:
            feat -= feat.mean(axis=0, keepdims=True)
        return feat.astype('float32')

    def __call__(self, y, sr):
        assert sr == 16000, "Support 16k wave only!"
        if len(y) > sr * 30:
            y = y[:int(sr*30)]  # truncate the maximum length of 30s.
        feat = self.extract_fbank(y, sr, cmn=True)
        feat = torch.from_numpy(feat).unsqueeze(0)
        feat = feat.float().to(self.device)
        self.model.eval()
        with torch.no_grad():
            embd = self.model.extractor(feat)
            rslt = self.model.forward(feat)
        embd = embd.squeeze(0).cpu().numpy()
        rslt = rslt.squeeze(0).cpu().numpy()
        return embd, rslt

def labeling(iii, args):
    isFirst = True
    
    model_dir = args.model_dir
    vad_path = args.vad_file
    output_dir = args.output_dir.strip('/') + '/'
    recording_dir = args.datadir.strip('/') + '/'
    
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    sv_extractor = SVExtractor(mdl_kwargs, fbank_kwargs, model_dir, device='cpu')
    with open(vad_path,'r') as f:
        vad_info = [line for line in f.read().split('\n') if line]
        
    with open('../emo/index/int2label.json','r') as f:
        identi = json.load(f)
        
    # loading vad results
    voiced_part = {}
    for line in tqdm(vad_info):
        line_info = line.split()
        start_time = float(line_info[3])
        end_time   = float(line_info[3]) + float(line_info[4])
        if not line_info[1] in voiced_part:
            voiced_part[line_info[1]] = []
        voiced_part[line_info[1]].append((start_time, end_time))
    
    recorders = os.listdir(recording_dir)
    all_recordings = []
    vad_result = {}
    for rec in recorders:
        all_recordings += [recording_dir+rec+'/'+r for r in os.listdir(recording_dir+rec) if r.endswith('.wav')]
    portion = len(all_recordings) // 4
    if i == 0:
        all_recordings = all_recordings[:portion]
    elif i == 1:
        all_recordings = all_recordings[portion:portion*2]
    elif i == 2:
        all_recordings = all_recordings[portion*2: portion*3]
    elif i == 3:
        all_recordings = all_recordings[portion*3:]
    
    # label the assigned portion
    for rcd in tqdm(all_recordings[:2], desc="process_%d"%os.getpid()):
        if isFirst:
            with open('kill_label.sh','a') as f:
                f.write('kill -9 %d\n'%os.getpid())
            isFirst = False
            
        if not rcd.split('/')[-1] in vad_result:
            vad_result[rcd.split('/')[-1]] = {}
#         rcd_path = recording_dir + recorder + '/' + rcd
        rcd_data, sr = librosa.load(rcd, sr=16000)
        try:
            rcd_voiced = voiced_part[rcd.split('/')[-1]]
            for (start_time, end_time) in rcd_voiced:
                if end_time - start_time <= 5:
                    tmp_clip = rcd_data[int(start_time * sr):int(end_time * sr)]
                    embd, result = sv_extractor(tmp_clip, sr)
                    probs = ["{0:0.4f}".format(i) for i in softmax(result)]
                    vad_result[rcd.split('/')[-1]]["(%.4f, %.4f)"%(start_time, end_time)] = dict(zip(list(identi.values()), probs))
                else:
                    time_shift = 1
                    curr_start = start_time
                    while end_time - curr_start >= 4:
                        curr_window = rcd_data[int(curr_start * sr):int((curr_start+5) * sr)]
                        embd, result = sv_extractor(curr_window, sr)
                        probs = ["{0:0.4f}".format(i) for i in softmax(result)]
                        vad_result[rcd.split('/')[-1]]["(%.4f, %.4f)"%(curr_start, curr_start+5)] = dict(zip(list(identi.values()), probs))
                        curr_start += 1
        except:
            with open('bad_file','a') as f:
                f.write(os.path.abspath(rcd)+'\n')
    
    with open(output_dir + 'recording_labels_process%d.json'%iii, 'w') as f:
        json.dump(vad_result, f)
    

if __name__ == "__main__":
    with open('kill_label.sh','w') as f:
        f.write('')
    with open('bad_file', 'w') as f:
        f.write('')
    
    args = parse_args()
    
    from multiprocessing import Process
    worker_count = 4
    worker_pool = []
    for i in range(worker_count):
        p = Process(target=labeling, args=(i, args))
        p.start()
        worker_pool.append(p)
    for p in worker_pool:
        p.join()  # Wait for all of the workers to finish.

    # Allow time to view results before program terminates.
    a = input("Finished")

