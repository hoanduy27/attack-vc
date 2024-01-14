import torch

class Attacker:
    def perturb(self, x: torch.Tensor):
        raise NotImplementedError

class TrainableAttacker(Attacker): 
    def fit_attacker(self, loader):
        raise NotImplementedError