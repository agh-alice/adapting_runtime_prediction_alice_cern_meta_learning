import torch
import torch.nn.functional as F

epsilon = 1e-8  # To avoid division by zero

def mae(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Calculate Mean Absolute Error (MAE).
    """
    return torch.mean(torch.abs(predictions - targets))

def mse(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Calculate Mean Squared Error (MSE).
    """
    return torch.mean((predictions - targets) ** 2)

def rmse(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Calculate Root Mean Squared Error (RMSE).
    """
    return torch.sqrt(mse(predictions, targets))

def mape(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    """
    return torch.mean(torch.abs((targets - predictions) / (targets + epsilon))) * 100

def uep(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Calculate Understimate Percentage (UEP).
    """
    underestimates = (predictions < targets).sum()
    total_predictions = predictions.numel()
    ue_percentage = (underestimates / total_predictions) * 100    
    return ue_percentage

def soft_uep(predictions: torch.Tensor, targets: torch.Tensor, sharpness=10.0) -> torch.Tensor:
    """
    Differentiable approximation of Underestimation Percentage.
    
    Returns a value in [0, 1] equal to the fraction of 
    predictions that are lower than targets.
    """
    # Soft indicator: ≈1 when pred < target, ≈0 when pred > target
    soft_under = torch.sigmoid((targets - predictions) * sharpness)
    
    return soft_under.mean()

def smape(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Calculate Symetric Mean Absolute Percentage Error (SMAPE)
    """
    return torch.mean(2 * (predictions - targets).abs() / (targets.abs() + predictions.abs() + epsilon)) * 100

def r2_score(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Calculate R-squared (coefficient of determination)
    """
    targets_mean = torch.mean(targets)

    ss_res = torch.sum((targets - predictions) ** 2)
    ss_tot = torch.sum((targets - targets_mean) ** 2)

    return 1.0 - ss_res / (ss_tot + epsilon)

def huber(predictions: torch.Tensor, targets: torch.Tensor, delta: float = 0.2) -> torch.Tensor:
    with torch.no_grad():
        return F.huber_loss(
            predictions,
            targets,
            delta=delta,
            reduction="mean",
        )