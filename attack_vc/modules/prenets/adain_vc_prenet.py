import torch
from torch import nn
import torchaudio 
import torchaudio.functional as F 
import torchaudio.transforms as T

from hyperpyyaml.core import load_hyperpyyaml

import pickle

from typing import Tuple


from attack_vc.modules.prenets.base import Prenet

class AdainVCPrenet(Prenet):
    def __init__(self, config: dict):
        # super(self, )
        super(AdainVCPrenet, self).__init__()
        self.args = config 
        self.config = config['preprocess']
        
        self.mean, self.std = self.load_attr()

        self.resampler = None

        self.melspectrogram_transformer = T.MelSpectrogram(
            # stft config
            n_fft = self.config["n_fft"],
            win_length = self.config["win_length"],
            hop_length = self.config["hop_length"],
            power = 1,
            pad_mode = "constant",

            # filterbanks config
            n_mels = self.config["n_mels"],
            sample_rate = self.config["sample_rate"],
            # n_stft= config["n_fft"] // 2 + 1,
            mel_scale = "slaney",
            norm = "slaney"
        )

    
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
    
    def wav_to_mel(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        speech_sr: torch.int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Arguments
            - speech: (B, C, T)
            - speech_lengths: (B,)
            - speech_sr: (B,)

        Returns
            - mel: (B, F, T_mel)
            - mel_lengths: (B,)
        """

        # (B, T)
        speech = speech[:, 0, :]

        # Init resampler
        if self.resampler is None:
            self.resampler = T.Resample(
                orig_freq = speech_sr, 
                new_freq = self.config["sample_rate"]
            )
        
        # Resample
        speech = self.resampler(speech)
    
        # Pre-emphasis
        speech =  torch.cat([
            speech[:, 0:1], 
            speech[:, 1:] - 0.99*speech[:, :-1]
        ], axis=-1)

        # Melspectrogram
        # (B, F, T_mel)
        speech = self.melspectrogram_transformer(speech)

        # Log mel
        speech[speech < 1.e-5] = 1.e-5
        speech = 20 * torch.log10(speech)
        
        # speech = (speech - self.config['ref_db'] + self.config['max_db']) / self.config['max_db']
        speech = torch.clamp(
            (speech - self.config['ref_db'] + self.config['max_db']) / self.config['max_db'], 
            1.e-8, 1.0
        )

        # Compute length in mel 
        # T_mel = floor(T_raw / hoplen) + 1
        speech_lengths = speech_lengths // self.config["hop_length"] + 1

        # Set mel values corresponding to padding position to zero
        speech[
            torch.cumsum(torch.ones_like(speech), -1) 
            > speech_lengths.unsqueeze(-1).unsqueeze(-1)
        ] = 0.0

        return speech, speech_lengths

    def normalize(self, mel, mel_lengths):
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
        mel = (mel - mean) / std
        
        mel[
            torch.cumsum(torch.ones_like(mel), -1) 
            > mel_lengths.unsqueeze(-1).unsqueeze(-1)
        ] = 0.0

        return mel, mel_lengths
    
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


    def forward(self, speech, speech_lengths, speech_sr):
        """
        Args:
            speech: (B, C, T)
            speech_lengths: (B, T)
            speech_sr: Int

        Returns:
            speech: (B, F, T_mel)
            speech_lengths: (B,)
        """

        speech, speech_lengths= self.wav_to_mel(
            speech, speech_lengths, speech_sr
        )

        speech, speech_lengths = self.normalize(speech, speech_lengths)
        
        return speech, speech_lengths

if __name__ == "__main__":
    prenet = AdainVCPrenet.from_config('/home/duy/github/attack-vc/model/config_adv.yaml')
    
    speech, sr = torchaudio.load("/home/duy/github/attack-vc/egs/adain-vc/exp_eps0.1/sample_per_utterance-attack_target/wav/p225/p225_008.wav")

    speech = speech.unsqueeze(0)

    speech_lengths = torch.tensor([speech.size(-1)])
    
    speech_srs = torch.tensor([sr])

    # speech=torch.rand((3, 1, 1000))

    # speech_lengths = torch.tensor([700,800,900])

    # speech_srs= torch.tensor([24000]*3)

    speech, speech_lengths = prenet(speech, speech_lengths, speech_srs)

    print(prenet.mean, prenet.std)

    print(speech)
    print(speech_lengths)
    print(speech.min())
    print(speech.max())