import torch


def loss_fn(outputs, targets):
    """
    Compute the Cross Entropy Loss.

    Args:
        outputs (torch.Tensor): Model predictions.
        targets (torch.Tensor): Ground truth labels.

    Returns:
        torch.Tensor: Computed loss.
    """
    return torch.nn.CrossEntropyLoss()(outputs, targets)
