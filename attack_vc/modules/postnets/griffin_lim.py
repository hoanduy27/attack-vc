import copy
import pickle
from typing import Tuple

import librosa
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from hyperpyyaml.core import load_hyperpyyaml
from torch import nn

import numpy as np
from scipy.signal import lfilter

from attack_vc.modules.postnets.base import PostNet


class GriffinLim(nn.Module):
    # def __init__(self, device):
    #     self.device = device
    def __init__(self, config: dict):
        # super(self, )
        super(GriffinLim, self).__init__()
        self.args = config 
        self.config = config['preprocess']
        
        self.mean, self.std = self.load_attr()
        self.init_config()

    def init_config(self):
        self.sample_rate = self.config["sample_rate"]
        self.preemph = self.config["preemph"]
        self.n_fft = self.config["n_fft"]
        self.hop_length = self.config["hop_length"]
        self.win_length = self.config["win_length"]
        self.n_mels = self.config["n_mels"]
        self.ref_db = self.config["ref_db"]
        self.max_db = self.config["max_db"]
        self.top_db = self.config["top_db"]
        self.n_iters = self.config["n_iters"]
        
    
    def load_attr(self):
        with open(self.args['feat_stats_path'], 'rb') as f:
            attr = pickle.load(f)

        mean = torch.from_numpy(attr['mean'])
        std = torch.from_numpy(attr['std'])

        return mean, std 
    
    @classmethod
    def from_config(cls, config_path):
        # if device is None:
        #     device = "cuda" if torch.cuda.is_available() else "cpu"
        # with open(config)
        with open(config_path, 'r') as f:
            config = load_hyperpyyaml(f)

        return cls(config)
    
    def denormalize(self, mel, mel_lengths):
        """
        Args:
            - mel (B, F, T_mel)

        Returns
            - mel (B, F, T_mel)
        """
        # mean: (F,) 
        # std:  (F,)

        mean = self.mean.unsqueeze(0).unsqueeze(-1)
        std = self.std.unsqueeze(0).unsqueeze(-1)

        # Set mel values corresponding to padding position to zero
        mel = mel * std + mean
        
        mel[
            torch.cumsum(torch.ones_like(mel), -1) 
            > mel_lengths.unsqueeze(-1).unsqueeze(-1)
        ] = 0.0

        return mel, mel_lengths
    
    def inv_mel_matrix(self):
        # F: stands for n_mels (for consistency)

        # (sr//2 + 1, F)
        m = F.melscale_fbanks(
            n_freqs = self.config["n_fft"]//2 + 1,
            n_mels = self.config["n_mels"], 
            f_min = 0,
            f_max = self.config["sample_rate"] // 2,
            sample_rate = self.config["sample_rate"],
            norm = 'slaney',
            mel_scale = 'slaney'
        )

        # (F, F)
        p = torch.matmul(m.T, m)

        # (F, )        
        d = [1.0 / x if torch.abs(x) > 1e-8 else x for x in torch.sum(p, axis=0)]

        # (F, F)
        return torch.matmul(m, torch.diag(torch.tensor(d)))

    def griffin_lim(self, mel):
        """
        Args:
            - mel (B, F, T_mel)
            - mel_length (B, T)
        """
        # Temporary solution
        
        X_best = copy.deepcopy(mel)
        for _ in range(self.n_iter):
            X_t = librosa.istft(
                X_best, 
                hop_length=self.hop_length, 
                win_length=self.win_length, 
                window="hann"
            )
            est = librosa.stft(
                X_t, 
                n_fft=self.n_fft, 
                hop_length=self.hop_length, 
                win_length=self.win_length
            )
            phase = est / np.maximum(1e-8, np.abs(est))
            X_best = mel * phase
        X_t = librosa.istft(
            X_best, 
            hop_length=self.hop_length, 
            win_length=self.win_length, 
            window="hann"
        )
        y = np.real(X_t)
        return torch.from_numpy(y)
    
    def forward(self, mel, mel_lengths):
        """
        Args:
            - mel (B, F, T_mel)
            - mel_lengths (B, T)

        Returns:
            - speech (B, T)
            - speech_lengths (B,)
        """
        # mel = mel.T

        mel = (torch.clamp(mel, 0.0, 1.0) * self.config["max_db"]) - self.config["max_db"] + self.config["ref_db"]

        mel = torch.pow(10.0, mel * 0.05)

        # (F, F)
        inv_mat = self.inv_mel_matrix()
        mag = torch.dot(inv_mat)
        
        # (F,F) x (B,F,T) -> (B, F, T)
        mag = torch.dot(inv_mat, mel)

        wav = self.griffin_lim(mag)

        wav = F.lfilter(wav)


    # def forward(mel, mel_lengths):
    #     """
    #     Args:
    #         - mel: (B, F, T_mel)
    #         - mel_lengths (B, T)

    #     Returns     
    #         - speech: (B, T)
    #         - speech_lengths: (B,)
    #     """        