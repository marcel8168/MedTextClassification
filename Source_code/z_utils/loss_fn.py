import torch


def loss_fn(outputs, targets):
    return torch.nn.CrossEntropyLoss()(outputs, targets)
