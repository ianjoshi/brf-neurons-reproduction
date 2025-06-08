import pathlib
from typing import Callable, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio.transforms import MFCC


class SpeechCommands(Dataset):
    """
    A PyTorch dataset loader for the Google Speech Commands v0.02 dataset.
    
    This class provides a convenient interface to load audio data from the
    Google Speech Commands dataset, with explicit train/validation/test splits. It wraps
    the torchaudio SPEECHCOMMANDS dataset and adds functionality for custom transforms
    and proper data splitting.

    Parameters:
    - root (str | pathlib.Path): Directory used to download or locate the dataset. If the dataset is not present
      and download=True, it will be downloaded to this location.
    - subset (str): Which data split to load. Must be one of {"training", "validation", "testing"}. 
      The splits are determined by the official validation_list.txt and testing_list.txt files.
    - transform (Callable[[torch.Tensor], torch.Tensor], optional): A function/transform that takes a waveform 
      tensor and returns a transformed version. Common transforms include MFCC, spectrogram, or mel spectrogram.
    - target_transform (Callable[[str], Any], optional): A function/transform that takes a label string and 
      returns a transformed version. Often used to convert string labels to numerical indices.
    - download (bool): If True, downloads the dataset from the internet and puts it in root directory.
      If dataset is already downloaded, it is not downloaded again. Defaults to True.

    Attributes:
    - transform (Callable or None): The transform applied to each waveform.
    - target_transform (Callable or None): The transform applied to each label.
    - _dataset (SPEECHCOMMANDS): The underlying torchaudio dataset instance.
    - _walker (list[pathlib.Path]): List of paths to audio files in the selected subset.
    """

    # Official split definition files provided with the dataset
    _VALIDATION_FILE = "validation_list.txt"
    _TEST_FILE = "testing_list.txt"

    def __init__(
        self,
        root: str | pathlib.Path,
        subset: str = "training",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
    ) -> None:
        print(f"\nInitializing SpeechCommands with subset={subset}")
        
        # Validate subset parameter
        if subset not in {"training", "validation", "testing"}:
            raise ValueError("subset must be 'training', 'validation', or 'testing'")

        # Ensure root directory exists before download
        root = pathlib.Path(root)
        root.mkdir(parents=True, exist_ok=True)
        print(f"Using root directory: {root}")

        # Initialize base dataset
        print("Loading base SPEECHCOMMANDS dataset...")
        self._dataset = SPEECHCOMMANDS(root=str(root), download=download)
        print("Base dataset loaded successfully")

        # Load official split definitions
        print("Loading split definitions...")
        val_list = self._load_split_list(self._VALIDATION_FILE)
        test_list = self._load_split_list(self._TEST_FILE)
        print(f"Found {len(val_list)} validation files and {len(test_list)} test files")

        # Build walker list based on selected subset
        if subset == "training":
            # Training set consists of all files not in validation or test sets
            exclude = set(val_list + test_list)
            self._walker = [p for p in self._dataset._walker if p not in exclude]
            print(f"Created training split with {len(self._walker)} files")
        elif subset == "validation":
            self._walker = val_list
            print(f"Created validation split with {len(self._walker)} files")
        else:  # "testing"
            self._walker = test_list
            print(f"Created testing split with {len(self._walker)} files")

        # Store transforms
        self.transform = transform
        self.target_transform = target_transform
        if transform:
            print("Waveform transform will be applied")
        if target_transform:
            print("Label transform will be applied")

    def _load_split_list(self, filename: str) -> list[pathlib.Path]:
        """
        Load and parse one of the official split definition files.

        Parameters:
        - filename (str): Name of the split definition file (e.g., "validation_list.txt")

        Returns:
        - list[pathlib.Path]: List of absolute paths to WAV files in the specified split.
        """
        txt_path = pathlib.Path(self._dataset._path) / filename
        print(f"Loading split list from {filename}")
        
        try:
            with txt_path.open() as f:
                paths = [pathlib.Path(self._dataset._path) / line.strip() for line in f]
            print(f"Successfully loaded {len(paths)} paths from {filename}")
            return paths
        except FileNotFoundError:
            print(f"Error: Split definition file not found: {txt_path}")
            raise
        except Exception as e:
            print(f"Error loading split list from {filename}: {str(e)}")
            raise

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.

        Returns:
        - int: Number of audio samples in the selected subset.
        """
        return len(self._walker)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        """
        Get a single sample from the dataset.

        Parameters:
        - index (int): Index of the sample to retrieve.

        Returns:
        - Tuple[torch.Tensor, str]: A tuple containing:
            - waveform: The audio waveform tensor
            - label: The corresponding label (string or transformed value)
        """
        try:
            # Load raw waveform and label from base dataset
            waveform, _, label, *_ = self._dataset[self._walker[index]]
            print(f"Loaded sample {index}: label={label}, waveform shape={waveform.shape}")

            # Apply optional transforms if specified
            if self.transform:
                waveform = self.transform(waveform)
                print(f"Applied waveform transform, new shape={waveform.shape}")
            if self.target_transform:
                label = self.target_transform(label)
                print(f"Applied label transform, new label={label}")

            return waveform, label
        except Exception as e:
            print(f"Error loading sample {index}: {str(e)}")
            raise


# Example usage
if __name__ == "__main__":
    # Create dataset instance
    train_ds = SpeechCommands(
        pathlib.Path("./google-speech-commands/data"),
        subset="training"
    )

    # Create DataLoader and inspect a batch
    loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    batch_wave, batch_label = next(iter(loader))
    print(f"\nSuccessfully loaded batch:")
    print(f"Waveform shape: {batch_wave.shape}")
    print(f"Labels: {batch_label}")
