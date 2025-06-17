import torch
import numpy as np

from dataclasses import dataclass
from torch import nn
from torch import Tensor


@dataclass
class LossConfig:
    lambda_bss: float
    lambda_gmgs: float
    score_mtx: torch.Tensor  # for GMGS


class Losser:
    def __init__(self, config: LossConfig, device: str):
        self.ce_loss = nn.CrossEntropyLoss().to(device)
        self.config = config
        self.device = device
        self.accum = []

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        Compute loss
        """
        loss = self.ce_loss(y_pred, torch.argmax(y_true, dim=1))
        # print("y_pred", y_pred.shape, "y_true", y_true.shape)
        gmgs_loss = self.calc_gmgs_loss(y_pred, y_true)
        if gmgs_loss.isnan():
            gmgs_loss = torch.tensor(0.0).to(self.device)
        bss_loss = self.calc_bss_loss(y_pred, y_true)
        if bss_loss.isnan():
            bss_loss = torch.tensor(0.0).to(self.device)
        loss = loss + \
            self.config.lambda_bss * bss_loss + \
            self.config.lambda_gmgs * gmgs_loss
        self.accum.append(loss.clone().detach().cpu().item())
        return loss

    def calc_gmgs_loss(self, y_pred: Tensor, y_true) -> Tensor:
        """
        Compute GMGS loss
        """
        score_mtx = torch.tensor(self.config.score_mtx).to(self.device)
        y_truel = torch.argmax(y_true, dim=1)
        weight = score_mtx[y_truel]
        py = torch.log(y_pred)
        output = torch.mul(y_true, py)
        output = torch.mul(output, weight)
        output = torch.mean(output)
        return -output

    def calc_bss_loss(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        Compute BSS loss
        """
        tmp = y_pred - y_true
        tmp = torch.mul(tmp, tmp)
        tmp = torch.sum(tmp, dim=1)
        tmp = torch.mean(tmp)
        return tmp

    def get_mean_loss(self) -> float:
        """
        Get mean loss
        """
        return np.mean(self.accum)

    def clear(self):
        """
        Clear accumulated loss
        """
        self.accum.clear()
