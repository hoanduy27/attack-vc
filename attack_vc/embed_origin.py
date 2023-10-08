import torch
from torch import nn 
from data_utils import load_model 
import pickle 
import yaml 
import os 
from attack_vc.models import AdaInVC
import data_utils as dutils
import numpy as np
from torch import Tensor
from tqdm import trange, tqdm
import sys
from data_utils import denormalize, file2mel, load_model, mel2wav, normalize
import pandas as pd 
import soundfile as sf

from attack_vc.attacker import Attacker


if __name__ == "__main__":
    tqdm.pandas()
    attacker = Attacker.from_dir("model")

    audio_dir = sys.argv[1]
    egs = sys.argv[2]
    attack_data = os.path.join(egs, 'data', sys.argv[3] + '.csv')
    # attack_config = os.path.join(egs, 'config', sys.argv[4] + '.yaml')

    exp_dir = os.path.join(egs, 'exp', f'{sys.argv[3]}-origin')
    spk_col = sys.argv[4]
    utt_col = sys.argv[5]
    
    os.makedirs(egs, exist_ok=True)
    os.makedirs(exp_dir, exist_ok=True)

    df = pd.read_csv(attack_data)

    def single_run(row):
        vc_tgt_name = os.path.basename(os.path.splitext(row[utt_col])[0])

        # mel_dir = os.path.join(exp_dir, 'mel', row.spk)
        emb_dir = os.path.join(exp_dir, 'emb_tgt', row[spk_col])
        # wav_dir = os.path.join(exp_dir, 'wav', row.spk)

        # os.makedirs(mel_dir, exist_ok=True)
        os.makedirs(emb_dir, exist_ok=True)
        # os.makedirs(wav_dir, exist_ok=True)

        vc_tgt = os.path.join(audio_dir, row[utt_col])
            
        emb = attacker.embed(vc_tgt)

        emb_file = os.path.join(emb_dir, vc_tgt_name + '.npy')

        np.save(emb_file, emb)
        
        # sf.write(wav_file, adv_wav, attacker.config["preprocess"]["sample_rate"])

                 
    df.progress_apply(
        single_run, axis=1
    )

