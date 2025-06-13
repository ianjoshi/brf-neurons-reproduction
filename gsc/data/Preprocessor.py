import torch
from typing import Tuple

class Preprocessor:
    """
    Preprocessing class for Google Speech Commands dataset batches.

    Parameters:
    - normalize_inputs (bool): If True, normalize input MFCC features (deprecated if precomputed).
    - permute_inputs (bool): If True, permute inputs and targets to [sequence_length, batch_size, ...].
    """
    def __init__(self, normalize_inputs: bool = False, permute_inputs: bool = True) -> None:
        self.normalize_inputs = normalize_inputs
        self.permute_inputs = permute_inputs

    def process_batch(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a batch of inputs and targets.

        Parameters:
        - inputs (torch.Tensor): Input tensor of shape [batch_size, sequence_length, input_size]
        - targets (torch.Tensor): Target tensor of shape [batch_size, sequence_length, num_classes]

        Returns:
        - Tuple[torch.Tensor, torch.Tensor]: Processed inputs and targets
        """
        if self.permute_inputs:
            inputs = inputs.permute(1, 0, 2)  # [sequence_length, batch_size, input_size]
            targets = targets.permute(1, 0, 2)  # [sequence_length, batch_size, num_classes]

        if self.normalize_inputs:
            mean = inputs.mean(dim=(0, 1), keepdim=True)
            std = inputs.std(dim=(0, 1), keepdim=True) + 1e-8
            inputs = (inputs - mean) / std

        return inputs, targets