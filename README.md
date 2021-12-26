# emoTag
emoTag helps you train emotion detection model for Chinese audios.
### Environment
```bash
pip install -r requirement.txt
```
### Data
We used [Emotional Speech Dataset (ESD) for Speech Synthesis and Voice Conversion](https://github.com/HLTSingapore/Emotional-Speech-Data) from HLT Singapore.

### Train Emotion Classifier
Use this command to train a classifier. Adjust training setups in `conf/logfbank_train-emo.json`.
```python
python train.py --config conf/logfbank_train-emo.json --name task_trial_1
```
Models and logs will be find in `exp/`.
```
usage: train.py [-h] [-c CONFIG] [-r RESUME] [-n NAME] [--lr LR] [--bs BS]
                [--train_utt2wav TRAIN_UTT2WAV] [--val_utt2wav VAL_UTT2WAV]
                [--blocks BLOCKS] [--optimizer OPTIMIZER]
                [--train_pad0 TRAIN_PAD0] [--devel_pad0 DEVEL_PAD0]
                [--pretrain PRETRAIN]

PyTorch Template

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        config file path (default: None)
  -r RESUME, --resume RESUME
                        path to latest checkpoint (default: None)
  -n NAME, --name NAME
  --lr LR, --learning_rate LR
  --bs BS, --batch_size BS
  --train_utt2wav TRAIN_UTT2WAV
  --val_utt2wav VAL_UTT2WAV
  --blocks BLOCKS
  --optimizer OPTIMIZER
  --train_pad0 TRAIN_PAD0
  --devel_pad0 DEVEL_PAD0
  --pretrain PRETRAIN
```

### Infer labels
```python
python infer_label.py
```
Adjust the `vad_file` param and code if necessary to adapt to new tasks. `infer_label.py` adopted multiprocessing, increased cpu utilities rate and inference efficiency. See usage details below.
```
usage: infer_label.py [-h] [--vad_file VAD_FILE] [--model_dir MODEL_DIR]
                      [--output_dir OUTPUT_DIR]

parse model info

optional arguments:
  -h, --help            show this help message and exit
  --vad_file VAD_FILE
  --model_dir MODEL_DIR
  --output_dir OUTPUT_DIR
```