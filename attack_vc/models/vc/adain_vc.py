from attack_vc.models.vc.base import VCModel

from hyperpyyaml.core import load_hyperpyyaml

from attack_vc.modules.prenets.base import Prenet
from attack_vc.modules.postnets.base import PostNet
from attack_vc.modules.nets.base import Net


class AdainVC(VCModel):
    def __init__(self, 
                 prenet: Prenet, 
                 net: Net, 
                 postnet: PostNet, 
                 config: dict
    ):
        self.prenet = prenet 
        self.net = net 
        self.postnet = postnet
    
        self.config = config 

    @classmethod 
    def from_config(cls, config_path):
        with open(config_path, 'r') as f:
            config = load_hyperpyyaml(f)

        print(config)

        arch_conf = config['architecture']

        prenet, net, postnet = None, None, None 

        if config.get('prenet', None) is not None:
            prenet_cls = arch_conf['prenet']
            prenet = prenet_cls.from_config(config_path)

        if config.get('net', None) is not None:
            net_cls = arch_conf['net']
            net = net_cls.from_config(config_path)

        if config.get('postnet', None) is not None:
            postnet_cls = arch_conf['postnet']
            postnet = postnet_cls.from_config(config_path)

        return cls(prenet, net, postnet, config)

    def forward(self, speech, speech_lengths, speech_sr):
        """
        Args:
            - speech: (B, C, T)
            - speech_lengths: (B, T)
            - speech_sr: Int

            It is important to make sure that every speech in the same batch must have the same sample rate

        Returns:

        """

        # mel_lengths is unused now, just keeping for future

        mel, mel_lengths = self.prenet(
            speech, 
            speech_lengths, 
            speech_sr
        )

        mu, log_sigma, emb, dec = self.net(
            mel
        )

        return mu, log_sigma, emb, dec 
    
    def inference(self, ref_speech, content_speech):
        pass 

    

class AdversarialAdainVC(AdainVC):
    def __init__(self, prenet, net, postnet, config):
        self.prenet = prenet 
        self.net = net 
        self.postnet = postnet
    
        self.config = config 

