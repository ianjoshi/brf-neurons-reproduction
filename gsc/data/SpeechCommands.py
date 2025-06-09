import pathlib
import os
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset
from torchaudio.datasets import SPEECHCOMMANDS

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
      returns a transformed version. Often used to convert string labels to numerical indices or one-hot encodings.
    - sequence_length (int): Desired fixed length of each sample sequence (for SNN input). Defaults to 1300.
    - download (bool): If True, downloads the dataset from the internet and puts it in root directory.
      If dataset is already downloaded, it is not downloaded again. Defaults to True.

    Attributes:
    - transform (Callable or None): The transform applied to each waveform.
    - target_transform (Callable or None): The transform applied to each label.
    - _dataset (SPEECHCOMMANDS): The underlying torchaudio dataset instance.
    - _walker (list[str]): List of relative paths to audio files in the selected subset.
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
        sequence_length: int = 1300,
        download: bool = True,
    ) -> None:
        print(f"\nInitializing SpeechCommands with subset={subset}")

        # Print available audio backends
        print(f"Available torchaudio backends: {torchaudio.list_audio_backends()}")
        
        # Validate subset parameter
        if subset not in {"training", "validation", "testing"}:
            raise ValueError("subset must be 'training', 'validation', or 'testing'")

        # Ensure root directory exists before download
        self.root = pathlib.Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        print(f"Using root directory: {self.root}")

        # Initialize base dataset
        print("Loading base SPEECHCOMMANDS dataset...")
        self._dataset = SPEECHCOMMANDS(root=str(self.root), download=download)
        print("Base dataset loaded successfully")

        # Load official split definitions
        print("Loading split definitions...")
        val_list = self._load_split_list(self._VALIDATION_FILE)
        test_list = self._load_split_list(self._TEST_FILE)
        print(f"Found {len(val_list)} validation files and {len(test_list)} test files")

        # Normalize dataset walker paths to use forward slashes and ensure relative paths
        dataset_root = pathlib.Path(self._dataset._path)
        dataset_walker = []
        for p in self._dataset._walker:
            try:
                # Try to make the path relative to dataset_root
                rel_path = str(pathlib.Path(p).relative_to(dataset_root).as_posix())
                full_path = dataset_root / rel_path
                if full_path.exists():
                    dataset_walker.append(rel_path)
                else:
                    print(f"Warning: File does not exist: {full_path}")
            except ValueError:
                # If relative_to fails, assume p is already relative
                rel_path = str(pathlib.Path(p).as_posix())
                full_path = dataset_root / rel_path
                if full_path.exists():
                    dataset_walker.append(rel_path)
                else:
                    print(f"Warning: File does not exist: {full_path}")

        print(f"Total files in dataset_walker: {len(dataset_walker)}")

        # Build walker list based on selected subset
        if subset == "training":
            # Training set consists of all files not in validation or test sets
            exclude = set(val_list + test_list)
            self._walker = [p for p in dataset_walker if p not in exclude]
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
        self.sequence_length = sequence_length

        if transform:
            print("Waveform transform will be applied")
        if target_transform:
            print("Label transform will be applied")

        if len(self._walker) == 0:
            raise ValueError(f"No files found for {subset} split. Check dataset integrity or split files.")

    def _load_split_list(self, filename: str) -> list[str]:
        """
        Load and parse one of the official split definition files.

        Parameters:
        - filename (str): Name of the split definition file (e.g., "validation_list.txt")

        Returns:
        - list[str]: List of relative paths to WAV files in the specified split.
        """
        txt_path = pathlib.Path(self._dataset._path) / filename
        print(f"Loading split list from {filename}")
        
        try:
            with txt_path.open() as f:
                # Normalize paths to use forward slashes and ensure they are relative
                paths = [str(pathlib.Path(line.strip()).as_posix()) for line in f if line.strip()]
            print(f"Successfully loaded {len(paths)} paths from {filename}")
            return paths
        except FileNotFoundError:
            print(f"Error: Split definition file not found: {txt_path}")
            raise
        except Exception as e:
            print(f"Error loading split list from {filename}: {str(e)}")
            raise

    def _fix_length(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pad or truncate the time dimension of a tensor to `self.sequence_length`.

        Parameters:
        - x (Tensor): Input tensor with shape [time, features].

        Returns:
        - Tensor: Output tensor with shape [sequence_length, features].
        """
        if x.size(0) > self.sequence_length:
            return x[:self.sequence_length]
        elif x.size(0) < self.sequence_length:
            pad_size = self.sequence_length - x.size(0)
            return F.pad(x, (0, 0, 0, pad_size))  # Pad on the time axis
        return x

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.

        Returns:
        - int: Number of audio samples in the selected subset.
        """
        return len(self._walker)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.

        Parameters:
        - index (int): Index of the sample to retrieve.

        Returns:
        - Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - waveform: The MFCC/padded tensor of shape [sequence_length, input_size]
            - label: One-hot or class label tensor of shape [sequence_length, num_classes]
        """
        try:
            # Get the relative path for this index
            rel_path = self._walker[index]
            
            # Construct the full path correctly
            full_path = str(pathlib.Path(self._dataset._path) / rel_path)
            print(f"Constructed full path: {full_path}")  # Debug print
            
            # Verify file exists
            if not pathlib.Path(full_path).exists():
                raise FileNotFoundError(f"Audio file not found: {full_path}")
            
            # Load the audio file directly using torchaudio
            waveform, _ = torchaudio.load(full_path, backend="soundfile")
            
            # Get the label from the parent directory name
            label = pathlib.Path(rel_path).parent.name
            
            print(f"Loaded sample {index}: label={label}, waveform shape={waveform.shape}")

            # Apply transform (e.g., MFCC)
            if self.transform:
                waveform = self.transform(waveform)  # shape: [1, n_mfcc, time]
                waveform = waveform.squeeze(0).transpose(0, 1)  # [time, n_mfcc]
                waveform = self._fix_length(waveform)
                print(f"Applied waveform transform, new shape={waveform.shape}")

            # Apply target transform (e.g., one-hot + repeat)
            if self.target_transform:
                label = self.target_transform(label)
                print(f"Applied label transform, new label shape={label.shape}")

            return waveform, label
        except Exception as e:
            print(f"Error loading sample {index}: {str(e)}")
            raise