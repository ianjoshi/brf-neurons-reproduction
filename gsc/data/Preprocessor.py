import torch
from typing import Tuple

class Preprocessor:
    """
    Preprocessing class for Google Speech Commands dataset batches.

    This class takes input and target tensors from the SpeechCommandsDataLoader,
    permutes them to the required format for SNN models, and applies optional
    transformations.

    Parameters:
    - normalize_inputs (bool): If True, normalize input MFCC features to zero mean
      and unit variance per batch.
    """

    def __init__(self, normalize_inputs: bool = False) -> None:
        self.normalize_inputs = normalize_inputs

    def process_batch(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a batch of inputs and targets.

        Parameters:
        - inputs (torch.Tensor): Input tensor of shape [batch_size, sequence_length, input_size]
        - targets (torch.Tensor): Target tensor of shape [batch_size, sequence_length, num_classes]

        Returns:
        - Tuple[torch.Tensor, torch.Tensor]: Processed inputs and targets with shape
          [sequence_length, batch_size, input_size] and [sequence_length, batch_size, num_classes]
        """
        # Permute inputs: [batch_size, sequence_length, input_size] -> [sequence_length, batch_size, input_size]
        inputs = inputs.permute(1, 0, 2)

        # Permute targets: [batch_size, sequence_length, num_classes] -> [sequence_length, batch_size, num_classes]
        targets = targets.permute(1, 0, 2)

        # Normalize inputs if enabled
        if self.normalize_inputs:
            # Compute mean and std across sequence_length and batch_size dimensions
            mean = inputs.mean(dim=(0, 1), keepdim=True)
            std = inputs.std(dim=(0, 1), keepdim=True) + 1e-8  # Avoid division by zero
            inputs = (inputs - mean) / std

        return inputs, targets