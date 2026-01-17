import torch
import torch.nn as nn
from alice_jobs_package.training.metrics import soft_uep

class RubberHuberLoss(nn.Module):
    def __init__(self, delta=1.0, reduction='mean', penalty_factor=2.0):
        """
        Custom implementation of the Huber loss.
        
        Args:
            delta (float): Threshold at which to change between squared and absolute loss.
            reduction (str): Specifies the reduction to apply to the output:
                             'none' | 'mean' | 'sum'
        """
        super().__init__()
        self.delta = delta
        self.reduction = reduction
        self.penalty_factor = penalty_factor

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = input - target
        abs_error = torch.abs(error)

        quadratic = 0.5 * error ** 2
        linear = self.delta * (abs_error - 0.5 * self.delta)

        base_loss = torch.where(abs_error <= self.delta, quadratic, linear)

        loss = torch.where(error >= 0, base_loss, self.penalty_factor * base_loss)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class HuberUepSumLoss(nn.Module):
    def __init__(self, loss_coeff, aux_loss_coeff, *args, **kwargs) -> None:
        super().__init__()
        self.hubber_loss = torch.nn.HuberLoss(delta = loss_coeff, *args, **kwargs)
        self.soft_uep = soft_uep
        self.aux_loss_coeff = aux_loss_coeff
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        huber = self.hubber_loss(input, target)
        uep = self.soft_uep(input, target).mean()
        
        return huber + uep * self.aux_loss_coeff

class HuberUepLoss(nn.Module):
    def __init__(self, loss_coeff, aux_loss_coeff, *args, **kwargs) -> None:
        super().__init__()
        self.hubber_loss = torch.nn.HuberLoss(delta = loss_coeff, *args, **kwargs)
        self.soft_uep = soft_uep
        self.aux_loss_coeff = aux_loss_coeff
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        huber = self.hubber_loss(input, target)
        uep = self.soft_uep(input, target).mean()
        
        return huber, uep * self.aux_loss_coeff

@staticmethod
def normalize_losses(loss):
    if isinstance(loss, (list, tuple)):
        return loss

    elif isinstance(loss, torch.Tensor):
        if loss.dim() == 0:
            return [loss]
    
    return loss

@staticmethod
def normalize_loss(loss: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
    if isinstance(loss, (list, tuple)):
        loss = torch.stack(loss)
    return loss