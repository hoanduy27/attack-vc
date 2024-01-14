from torch import nn 
import yaml

from typing import Dict

from attack_vc.modules.nets.base import Net
from attack_vc.modules.postnets.base import PostNet
from attack_vc.modules.nets.base import Net



class VCModel(nn.Module):
    def __init__(self, prenet, net, postnet, config):
        self.prenet = prenet
        self.net = net 
        self.postnet = postnet 
        self.config = config 


    def forward(self, x):
        pass

    # def inference(self, batch):
    #     src, _, tgt, _ = self.prenet(batch)
    #     out_mel = self.inference(src, tgt)
    #     out_wav = self.postnet(out_mel)

    #     return out_wav
    
class AdversarialVCModel(VCModel):
    def __init__(self, prenet=None, net=None, postnet=None, attacker=None ):
        self.prenet = prenet 
        self.net = net 
        self.postnet = postnet 
        self.attacker = attacker 

    def attack(self, batch):
        pass 