import torch
from torch import nn 
from data_utils import load_model 
import pickle 
import yaml 
import os 
from attack_vc.models import AdaInVC
import data_utils as dutils
import numpy as np
import umap



class EmbeddingExtractor:
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
    embedder = EmbeddingExtractor.from_dir("model")

    log_dir = "logs/ESD"
    data_dir = "/olli_data/data/ESD"

    os.makedirs(log_dir, exist_ok=True)
    for root, dirs, files in os.walk(data_dir):
        # for dirname in dirs:
        #     new_dir = os.path.join(log_dir, os.path.relpath(root, data_dir), dirname)
        #     os.makedirs(new_dir, exist_ok=True)

        for filename in files:
            if filename.endswith(".wav"):
                filepath = os.path.join(root, filename)
                
                new_dir = os.path.join(
                    log_dir,
                    os.path.relpath(os.path.dirname(filepath), data_dir)
                )

                # Make feat dir if not exist
                os.makedirs(new_dir, exist_ok=True)

                # Path to save feat corresponding to wav file
                feat_savepath = os.path.join(new_dir, os.path.splitext(filename)[0] + ".npy")

                # feature extract
                feat = embedder.embed(filepath)
                
                np.save(feat_savepath, feat)

                print(f"{filepath} => {feat_savepath}")