import torch
import torch.nn.functional as F

class OneHotTargetTransform:
    """
    Converts label strings into repeated one-hot encodings across a sequence.

    Parameters:
    - label_to_index (dict): Mapping from label string to integer class index.
    - sequence_length (int): Desired output sequence length.
    - num_classes (int): Total number of label classes.
    """
    def __init__(self, label_to_index: dict, sequence_length: int, num_classes: int):
        self.label_to_index = label_to_index
        self.sequence_length = sequence_length
        self.num_classes = num_classes

    def __call__(self, label: str) -> torch.Tensor:
        index = self.label_to_index[label]
        one_hot = F.one_hot(torch.tensor(index), num_classes=self.num_classes).float()
        return one_hot.repeat(self.sequence_length, 1)
