import torch
import typing
import time


def apply_seq_loss(
    criterion: torch.nn.Module,
    outputs: torch.Tensor,
    targets: torch.Tensor,
    scale_func: typing.Callable[[int], float] = None
) -> torch.Tensor:
    """
    Apply loss function over a sequence of outputs and targets.

    Parameters:
    - criterion (torch.nn.Module): Loss function (e.g., NLLLoss).
    - outputs (torch.Tensor): Model outputs of shape [sequence_length, batch_size, num_classes].
    - targets (torch.Tensor): One-hot target labels of shape [sequence_length, batch_size, num_classes].
    - scale_func (Callable[[int], float], optional): Scaling function for each time step.

    Returns:
    - torch.Tensor: Accumulated loss over the sequence.
    """
    # Shape: [sequence_length, batch_size, num_classes]
    sequence_length = outputs.shape[0]

    # Get class indices from one-hot targets: [sequence_length, batch_size]
    targets_argmax = targets.argmax(dim=2)

    # Apply log softmax for NLLLoss
    log_softmax = torch.nn.LogSoftmax(dim=2)
    out = log_softmax(outputs)

    loss = 0

    if scale_func is None:
        for t in range(sequence_length):
            loss += criterion(out[t], targets_argmax[t])
    else:
        for t in range(sequence_length):
            loss += scale_func(t) * criterion(out[t], targets.argmax[t])

    return loss


def count_correct_prediction(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> int:
    """
    Count the number of correct predictions in a sequence.

    Parameters:
    - predictions (torch.Tensor): Model outputs of shape [sequence_length, batch_size, num_classes].
    - targets (torch.Tensor): One-hot target labels of shape [sequence_length, batch_size, num_classes].

    Returns:
    - int: Number of correct predictions across the sequence and batch.
    """
    # Compare argmax of predictions and targets: [sequence_length, batch_size]
    return predictions.argmax(dim=2).eq(targets.argmax(dim=2)).sum().item()


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