import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from models.score_net import Score_Net
from utils.r3_diffuser import R3Diffuser 
from utils.so3_diffuser import SO3Diffuser 

#----------------------------------------------------------------------------
# Main wrapper for inference

class Score_Model(pl.LightningModule):
    def __init__(
        self,
        model,
        diffuser,
        experiment,
    ):
        super().__init__()
        
        # translation
        self.perturb_tr = experiment.perturb_tr

        # rotation
        self.perturb_rot = experiment.perturb_rot

        # diffuser
        if self.perturb_tr:
            self.r3_diffuser = R3Diffuser(diffuser.r3)
        if self.perturb_rot:
            self.so3_diffuser = SO3Diffuser(diffuser.so3)

        # net
        self.net = Score_Net(model)
        print(model)
    
    def forward(self, batch):
        # grab some input 
        rec_pos = batch["rec_pos"]
        lig_pos = batch["lig_pos"]

        # move the lig center to origin
        center = lig_pos[..., 1, :].mean(dim=0)
        rec_pos -= center
        lig_pos -= center

        # predict
        outputs = self.net(batch, predict=True)

        return outputs
