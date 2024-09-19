# The code is adopted from:
# https://github.com/jasonkyuyim/se3_diffusion/blob/master/data/r3_diffuser.py

import numpy as np
import torch

#----------------------------------------------------------------------------
# Helper functions

move_to_np = lambda x: x.cpu().detach().numpy()

#----------------------------------------------------------------------------
# VE-SDE diffuser class for diffusion in 3D translational (R(3)) space 

class R3Diffuser:
    def __init__(self, conf):
        self.min_sigma = conf.min_sigma
        self.max_sigma = conf.max_sigma

    def sigma(self, t):
        return self.min_sigma * (self.max_sigma / self.min_sigma) ** t

    def diffusion_coef(self, t):
        return self.sigma(t) * np.sqrt(2 * (np.log(self.max_sigma) - np.log(self.min_sigma))) 

    def torch_score(self, tr_t, t):
        return -tr_t / self.sigma(t)**2

    def score_scaling(self, t: float):
        return 1 / self.sigma(t)

    def forward_marginal(self, t: float):
        if not np.isscalar(t):
            raise ValueError(f'{t} must be a scalar.')
        z = np.random.randn(1, 3)
        tr_t = self.sigma(t) * z 
        tr_score = self.torch_score(tr_t, t)
        return tr_t, tr_score

    def torch_reverse(
            self,
            score_t: torch.tensor,
            dt: torch.tensor,
            t: float,
            noise_scale: float=1.0,
            ode: bool=False,
        ):
        if not np.isscalar(t): raise ValueError(f'{t} must be a scalar.')
        g_t = self.diffusion_coef(t)
        if not ode:
            z = noise_scale * torch.randn(1, 3, device=score_t.device)
            perturb = (g_t ** 2) * score_t * dt + g_t * torch.sqrt(dt) * z
        else:
            perturb = 0.5 * (g_t ** 2) * score_t * dt
        return perturb.float()

