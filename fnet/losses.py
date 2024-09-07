"""Loss functions for fnet models."""


from typing import Optional

import torch
from torch.nn import functional as F

from pytorch3dunet.unet3d.losses import DiceLoss, BCEDiceLoss


class HeteroscedasticLoss(torch.nn.Module):
    """Loss function to capture heteroscedastic aleatoric uncertainty."""

    def forward(self, y_hat_batch: torch.Tensor, y_batch: torch.Tensor):
        """Calculates loss.

        Parameters
        ----------
        y_hat_batch
           Batched, 2-channel model output.
        y_batch
           Batched, 1-channel target output.

        """
        mean_batch = y_hat_batch[:, 0:1, :, :, :]
        log_var_batch = y_hat_batch[:, 1:2, :, :, :]
        loss_batch = (
            0.5 * torch.exp(-log_var_batch) * (mean_batch - y_batch).pow(2)
            + 0.5 * log_var_batch
        )
        return loss_batch.mean()


class WeightedMSE(torch.nn.Module):
    """Criterion for weighted mean-squared error."""

    def forward(
        self,
        y_hat_batch: torch.Tensor,
        y_batch: torch.Tensor,
        weight_map_batch: Optional[torch.Tensor] = None,
    ):
        """Calculates weighted MSE.

        Parameters
        ----------
        y_hat_batch
            Batched prediction.
        y_batch
            Batched target.
        weight_map_batch
            Optional weight map.

        """
        if weight_map_batch is None:
            return torch.nn.functional.mse_loss(y_hat_batch, y_batch)
        assert torch.all((weight_map_batch == 0) | (weight_map_batch == 1)), "Weight map must be binary"
        return (weight_map_batch * (y_hat_batch - y_batch) ** 2).mean()


class WMSEDiceLoss(torch.nn.Module):
    """Linear combination of MSE and Dice losses on the same prediction."""

    def __init__(self, gamma, k):
        super(WMSEDiceLoss, self).__init__()
        self.k = k
        self.gamma = gamma
        self.dice = DiceLoss(normalization="none")

    def tune_sigm(self, x, k):
        denominator = k - 2 * k * torch.abs(x) + 1
        result = (x - k * x) / denominator.clamp(min=torch.finfo(torch.float32).eps)
        return (result + 1) / 2

    def forward(
        self,
        y_hat_batch: torch.Tensor,
        y_batch: torch.Tensor,
        # weight_map_batch: Optional[torch.Tensor] = None,
        weight_map_batch: torch.Tensor
    ):
        # if weight_map_batch is None:
        #     mse_loss = F.mse_loss(y_hat_batch, y_batch)
        #     dice_loss = self.dice(self.tune_sigm(y_hat_batch, self.k), self.tune_sigm(y_batch, self.k))
        # else:
        
        # import pdb; pdb.set_trace()
        assert torch.all((weight_map_batch == 0) | (weight_map_batch == 1)), "Weight map must be binary"
        # TODO: consider target clipping, to minimum or just -1?
        mse_loss = (weight_map_batch * (y_hat_batch - y_batch) ** 2).mean()
        # mse_loss = ((y_hat_batch - torch.where(weight_map_batch == 1, y_batch, y_batch.min())) ** 2).mean() # minmsedice
        # TODO: consider tune_sigm on FG or on BG only
        # self.dice(self.tune_sigm(y_hat_batch*weight_map_batch, self.k), self.tune_sigm(y_batch*weight_map_batch, self.k))
        dice_loss = self.dice(self.tune_sigm(y_hat_batch, self.k), weight_map_batch)
        # dice_loss = self.dice(self.tune_sigm(y_hat_batch*(1-weight_map_batch), self.k), weight_map_batch) # wmsedicebg

        return (1 - self.gamma) * dice_loss + self.gamma * mse_loss
    