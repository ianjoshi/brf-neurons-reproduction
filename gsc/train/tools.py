import torch
import typing
import time
import torch.nn.functional as F

def apply_seq_loss(
    criterion: torch.nn.Module,
    outputs: torch.Tensor,
    targets: torch.Tensor,
    scale_func: typing.Callable[[int], float] = None
) -> torch.Tensor:
    """
    Apply loss function for sequence-level classification using the last time step.

    Parameters:
    - criterion (torch.nn.Module): Loss function (e.g., NLLLoss).
    - outputs (torch.Tensor): Model outputs of shape [sequence_length, batch_size, num_classes].
    - targets (torch.Tensor): One-hot target labels of shape [sequence_length, batch_size, num_classes].
    - scale_func (Callable[[int], float], optional): Not used for sequence-level classification.

    Returns:
    - torch.Tensor: Loss for the sequence.
    """
    # Use last time step: outputs[-1] is [batch_size, num_classes]
    out = F.log_softmax(outputs[-1], dim=1)  # Apply log_softmax along class dimension

    # Get class indices from one-hot targets for last time step: [batch_size]
    target_indices = targets[-1].argmax(dim=1)

    # Compute loss
    loss = criterion(out, target_indices)

    return loss


def count_correct_prediction(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> int:
    """
    Count the number of correct predictions for sequence-level classification.

    Parameters:
    - predictions (torch.Tensor): Model outputs of shape [sequence_length, batch_size, num_classes].
    - targets (torch.Tensor): One-hot target labels of shape [sequence_length, batch_size, num_classes].

    Returns:
    - int: Number of correct predictions for the batch.
    """
    # Use last time step: [batch_size]
    pred = predictions[-1].argmax(dim=1)
    targ = targets[-1].argmax(dim=1)
    return pred.eq(targ).sum().item()


class PerformanceCounter:
    """
    A simple performance counter for measuring elapsed time.
    """
    def __init__(self) -> None:
        self.start_time = 0

    def reset(self) -> None:
        self.start_time = time.perf_counter()

    def time(self) -> float:
        current_time = time.perf_counter()
        return current_time - self.start_time