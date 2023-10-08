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
class Attacker:
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

    

    def emb_attack(
            self, 
            vc_tgt: Tensor, 
            adv_tgt: Tensor, 
            eps: float, 
            n_iters: int,
            untarget_strength : float = 0.1
    ):  
        assert untarget_strength == 1 or adv_tgt is not None

        ptb = torch.zeros_like(vc_tgt).normal_(0, 1).requires_grad_(True)
        opt = torch.optim.Adam([ptb], lr=5e-3)
        criterion = nn.MSELoss()
        pbar = trange(n_iters)

        with torch.no_grad():
            org_emb = self.model.speaker_encoder(vc_tgt)
            if adv_tgt is not None:
                tgt_emb = self.model.speaker_encoder(adv_tgt)
            else:
                tgt_emb = None 

        for _ in pbar:
            adv_inp = vc_tgt + eps * ptb.tanh()
            adv_emb = self.model.speaker_encoder(adv_inp)
            
            loss_untargeted = criterion(adv_emb, org_emb)

            if tgt_emb is not None: 
                loss_targeted = criterion(adv_emb, tgt_emb)
            else:
                loss_targeted = None 

            if loss_targeted:
                loss = (
                    (1 - untarget_strength) * loss_targeted 
                    - untarget_strength * loss_untargeted
                )
            else:
                loss = -loss_untargeted

            loss_info = dict(
                loss_untargeted = loss_untargeted.item(),
                loss_targeted = loss_targeted.item() if loss_targeted else None 
            )
            # loss = -criterion(adv_emb, org_emb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            description = f'loss={loss.item()}, norm_ptb={ptb.norm()}'
            for loss_name,val in loss_info.items():
                if val:
                    description += f", {loss_name}={val}"

            pbar.set_description(description)

        with torch.no_grad():
            output_emb = self.model.speaker_encoder(vc_tgt + eps * ptb.tanh())  
        return vc_tgt + eps * ptb.tanh(), output_emb
    
    def embed_attack_file(self, vc_tgt, adv_tgt, eps, n_iters, untarget_strength):
        assert untarget_strength == 1 or adv_tgt is not None

        vc_tgt = file2mel(vc_tgt, **self.config["preprocess"])
        vc_tgt = normalize(vc_tgt, self.attr)
        vc_tgt = torch.from_numpy(vc_tgt).T.unsqueeze(0).to(self.device)

        if adv_tgt:
            adv_tgt = file2mel(adv_tgt, **self.config["preprocess"])
            adv_tgt = normalize(adv_tgt, self.attr)
            adv_tgt = torch.from_numpy(adv_tgt).T.unsqueeze(0).to(self.device)
        else: 
            adv_tgt = None 

        return self.emb_attack(vc_tgt, adv_tgt, eps, n_iters, untarget_strength )

    def attack(self, vc_tgt, adv_tgt, vc_src = None):
        attack_config = self.config['attack_config']
        attack_type = attack_config.get("attack_type", "emb")

        assert attack_type == "emb" or vc_src is not None
    
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
    attacker = Attacker.from_dir("model")

    audio_dir = sys.argv[1]
    egs = sys.argv[2]
    attack_data = os.path.join(egs, 'data', sys.argv[3] + '.csv')
    attack_config = os.path.join(egs, 'config', sys.argv[4] + '.yaml')

    exp_dir = os.path.join(egs, 'exp', f'{sys.argv[3]}-{sys.argv[4]}')
    


    with open(attack_config, 'r') as f:
        attack_config = yaml.load(f, Loader = yaml.Loader)
    
    os.makedirs(egs, exist_ok=True)
    os.makedirs(exp_dir, exist_ok=True)

    df = pd.read_csv(attack_data)

    def single_run(row):
        vc_tgt_name = os.path.basename(os.path.splitext(row.utt)[0])

        mel_dir = os.path.join(exp_dir, 'mel', row.spk)
        emb_dir = os.path.join(exp_dir, 'emb', row.spk)
        wav_dir = os.path.join(exp_dir, 'wav', row.spk)

        mel_file = os.path.join(mel_dir, vc_tgt_name + '.npy')
        emb_file = os.path.join(emb_dir, vc_tgt_name + '.npy')
        wav_file = os.path.join(wav_dir, vc_tgt_name + '.wav')

        if os.path.exists(emb_file):
            return

        os.makedirs(mel_dir, exist_ok=True)
        os.makedirs(emb_dir, exist_ok=True)
        os.makedirs(wav_dir, exist_ok=True)

        vc_tgt = os.path.join(audio_dir, row.utt)

        if 'adv_utt' in row:
            adv_tgt = os.path.join(audio_dir, row.adv_utt)
        else:
            adv_tgt = None 
            
        adv_mel, adv_emb = attacker.embed_attack_file(
            vc_tgt, 
            adv_tgt,
            **attack_config
        )

        adv_wav = adv_mel.squeeze(0).T
        adv_wav = denormalize(adv_wav.data.cpu().numpy(), attacker.attr)
        adv_wav = mel2wav(adv_wav, **attacker.config["preprocess"])

        print(adv_mel.shape)
        print(adv_emb.shape)

        

        np.save(mel_file, adv_mel.squeeze(0).cpu().detach().numpy())
        np.save(emb_file, adv_emb.squeeze(0).cpu().detach().numpy())
        
        sf.write(wav_file, adv_wav, attacker.config["preprocess"]["sample_rate"])

                 
    df.progress_apply(
        single_run, axis=1
    )

