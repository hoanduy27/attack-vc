import torch
from torch import nn 
from data_utils import load_model 
import pickle 
import yaml 
import os 
from models import AdaInVC
import data_utils as dutils
import numpy as np
from torch import Tensor
from tqdm import trange, tqdm
import sys
from data_utils import denormalize, file2mel, load_model, mel2wav, normalize
import pandas as pd 
import soundfile as sf
class Inferencer:
    def __init__(
        self,
        model: AdaInVC, 
        config: dict, 
        attr: dict, 
        device: str
    ):
        self.model = model
        self.config = config 
        self.attr = attr 
        self.device = device 


    def embed(self, filepath):
        mel = dutils.file2mel(filepath, **self.config["preprocess"])
        mel = dutils.normalize(mel, self.attr)
        mel = torch.from_numpy(mel).T.unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model.speaker_encoder(mel)
        
        return emb.squeeze(0).cpu().numpy()

    def inference(self, source, target, output):
        src_mel = file2mel(source, **self.config["preprocess"])
        tgt_mel = file2mel(target, **self.config["preprocess"])

        src_mel = normalize(src_mel, self.attr)
        tgt_mel = normalize(tgt_mel, self.attr)

        src_mel = torch.from_numpy(src_mel).T.unsqueeze(0).to(self.device)
        tgt_mel = torch.from_numpy(tgt_mel).T.unsqueeze(0).to(self.device)

        with torch.no_grad():
            out_mel = self.model.inference(src_mel, tgt_mel)
            out_mel = out_mel.squeeze(0).T
        out_mel = denormalize(out_mel.data.cpu().numpy(), self.attr)
        out_wav = mel2wav(out_mel, **self.config["preprocess"])

        sf.write(output, out_wav, self.config["preprocess"]["sample_rate"])

    @classmethod 
    def from_dir(cls, model_dir: str, device: str = None ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        attr_path = os.path.join(model_dir, "attr.pkl")
        cfg_path = os.path.join(model_dir, "config.yaml")
        model_path = os.path.join(model_dir, "model.ckpt")

        attr = pickle.load(open(attr_path, "rb"))
        config = yaml.safe_load(open(cfg_path, "r"))
        model = AdaInVC(config["model"]).to(device)
        model.load_state_dict(torch.load(model_path))

        return cls(model, config, attr, device)

if __name__ == "__main__":
    tqdm.pandas()
    inferencer = Inferencer.from_dir("model")

    def_audio_dir = sys.argv[1]
    content_audio_dir = sys.argv[2]
    egs = sys.argv[3]
    inference_data = os.path.join(egs, 'data', sys.argv[4] + '.csv')
    
    exp_dir = os.path.join(egs, 'exp', os.path.basename(def_audio_dir))
    
    os.makedirs(egs, exist_ok=True)
    # os.makedirs(exp_dir, exist_ok=True)

    df = pd.read_csv(inference_data)

    def single_run(row):
        def_utt = os.path.join(def_audio_dir, 'wav', row.utt)
        content_utt = os.path.join(content_audio_dir, row.content_utt)
        output = os.path.join(exp_dir, 'wav_syn', row.utt)

        os.makedirs(os.path.join(exp_dir, 'wav_syn', row.spk), exist_ok=True)

        if os.path.exists(output):
            return

        inferencer.inference(content_utt, def_utt, output)


    df.progress_apply(
        single_run, axis=1
    )

