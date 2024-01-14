import os 

import torch
import torchaudio
from torch import nn 
from torch.utils.data import Dataset, DataLoader

import pandas as pd 

class CsvDataset(Dataset):
    def __init__(self, csv_path, audio_root=None):
        self.csv_path = csv_path
        self.df = pd.read_csv(self.csv_path) 
        self.audio_root = audio_root 

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        assert idx < len(self)

        data = self.df.iloc[idx]
        data['audio_root'] = self.audio_root

        return data.to_dict()


def collate_fn(batch):
    ys = []
    y_lengths = []
    srs = []
    for data in batch: 
        y, sr = torchaudio.load(data['utt'])
        
        y_length = y.size(-1)
        
        y_lengths.append(y_length)  

        ys.append(y)

        srs.append(sr)

    y_lengths = torch.tensor(y_lengths)
    srs = torch.tensor(srs)

    max_length = torch.max(y_lengths)

    ys = torch.stack(
        [nn.functional.pad(
            y, (0, max_length - y.size(-1)), value=0.
        ) for y in ys]
    )

    return ys, srs

# def collate_fn(batch):
#     def_spk = []
#     def_speech = []
#     def_speech_lengths = []
#     adv_spk = []
#     adv_speech = []
#     adv_speech_lengths = [] 
#     torchaudio.load()

#     for data in batch:
#         def_spk.append(data['spk'])
#         def_path = os.path.join(data['audio_root'], data['utt'])\
#               if data['audio_root'] else data['utt']
        


#         if adv_spk is not None:
#             if 'adv_spk' in data:
#                 adv_spk.append(data['adv_spk'])
#                 adv_path = os.path.join(data['audio_root'], data['adv_utt'])\
#                       if data['audio_root'] else data['adv_utt']
#             else:
#                 adv_spk = None 
#                 adv_speech = None
#                 adv_speech_lengths = None 

#     return def_spk, def_speech, def_speech_lengths, adv_spk, adv_speech, adv_speech_lengths

if __name__ == "__main__":
    ds = CsvDataset('/home/duy/github/attack-vc/egs/adain-vc/data/sample_one_to_one.csv')

    dl = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_fn)

    a = next(iter(dl))

    
