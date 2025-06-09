import os
import sys
import pathlib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from typing import Tuple

from torch.utils.data import DataLoader
from torchaudio.transforms import MFCC

from gsc.data.SpeechCommands import SpeechCommands
from gsc.data.OneHotTargetTransform import OneHotTargetTransform

class SpeechCommandsDataLoader:
    """
    A class that sets up and returns PyTorch DataLoaders for the SpeechComsmands dataset,
    formatted for Spiking Neural Network (SNN) training.

    Parameters:
    - root (str): Root directory for the dataset.
    - sequence_length (int): Number of time steps to pad/truncate each sample to.
    - batch_size (int): Batch size for all splits.
    - num_workers (int): Number of worker threads for loading.
    - pin_memory (bool): Whether to pin memory (for CUDA optimization).
    """

    def __init__(
        self,
        root: str,
        sequence_length: int = 1300,
        batch_size: int = 4,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        self.root = root
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Build label-to-index map from the dataset
        print("Building label-to-index map...")
        base_ds = SpeechCommands(root=root, subset="training", download=True)
        
        # Extract unique labels from the dataset
        self.labels = sorted(set(pathlib.Path(path).parent.name for path in base_ds._walker))
        self.label_to_index = {label: idx for idx, label in enumerate(self.labels)}
        self.num_classes = len(self.labels)
        print(f"Detected {self.num_classes} unique labels")

        # Shared transforms
        self.mfcc_transform = MFCC(
            sample_rate=16000,
            n_mfcc=13,
            log_mels=True,
            melkwargs={"n_mels": 40, "n_fft": 400})
        
        self.label_transform = OneHotTargetTransform(
            label_to_index=self.label_to_index,
            sequence_length=self.sequence_length,
            num_classes=self.num_classes
        )

    def _create_loader(self, subset: str) -> DataLoader:
        """
        Internal helper to create a DataLoader for a given subset.
        """
        dataset = SpeechCommands(
            root=self.root,
            subset=subset,
            transform=self.mfcc_transform,
            target_transform=self.label_transform,
            sequence_length=self.sequence_length,
            download=False
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(subset == "training"),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )

    def get_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Returns:
        - Tuple of (train_loader, val_loader, test_loader)
        """
        print("Creating train/val/test loaders...")
        train_loader = self._create_loader("training")
        val_loader = self._create_loader("validation")
        test_loader = self._create_loader("testing")
        print("All loaders created successfully.")
        return train_loader, val_loader, test_loader
    
if __name__ == "__main__":
    # Instantiate the loader class
    loader_factory = SpeechCommandsDataLoader(
        root="./gcs-experiments/data",
        sequence_length=1300,
        batch_size=4,
        num_workers=2,
        pin_memory=False
    )

    # Get the DataLoaders
    train_loader, val_loader, test_loader = loader_factory.get_loaders()

    # Load one batch from the training set
    inputs, targets = next(iter(train_loader))

    print("\n✅ Example batch loaded:")
    print(f"Input shape (waveforms): {inputs.shape}")  # [batch_size, seq_len, n_mfcc]
    print(f"Target shape (one-hot):  {targets.shape}")  # [batch_size, seq_len, num_classes]