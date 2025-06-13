import pathlib
import os
from typing import Callable, Optional, Tuple
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset
from torchaudio.datasets import SPEECHCOMMANDS
from tqdm import tqdm

class SpeechCommands(Dataset):
    """
    A PyTorch dataset loader for the Google Speech Commands v0.02 dataset with in-memory caching.
    
    Parameters:
    - root (str | pathlib.Path): Directory used to download or locate the dataset.
    - subset (str): Which data split to load ("training", "validation", or "testing").
    - transform (Callable[[torch.Tensor], torch.Tensor], optional): Transform for waveforms (e.g., MFCC).
    - target_transform (Callable[[str], Any], optional): Transform for labels (e.g., one-hot encoding).
    - sequence_length (int): Desired fixed length of each sample sequence. Defaults to 1300.
    - download (bool): If True, downloads the dataset if not present. Defaults to True.
    - cache_data (bool): If True, caches transformed data in memory. Defaults to True.
    - preload_cache (bool): If True, preloads all data into cache during initialization. Defaults to False.
    """
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
        cache_data: bool = True,
        preload_cache: bool = False,
    ) -> None:
        print(f"\nInitializing SpeechCommands with subset={subset}, cache_data={cache_data}, preload_cache={preload_cache}")
        self.cache_data = cache_data
        self.cache = {} if cache_data else None  # In-memory cache for (waveform, label) tuples

        # Print available audio backends
        print(f"Available torchaudio backends: {torchaudio.list_audio_backends()}")
        
        # Validate subset parameter
        if subset not in {"training", "validation", "testing"}:
            raise ValueError("subset must be 'training', 'validation', or 'testing'")

        # Ensure root directory exists
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

        # Normalize dataset walker paths
        dataset_root = pathlib.Path(self._dataset._path)
        dataset_walker = []
        for p in self._dataset._walker:
            try:
                rel_path = str(pathlib.Path(p).relative_to(dataset_root).as_posix())
                full_path = dataset_root / rel_path
                if full_path.exists():
                    dataset_walker.append(rel_path)
                else:
                    print(f"Warning: File does not exist: {full_path}")
            except ValueError:
                rel_path = str(pathlib.Path(p).as_posix())
                full_path = dataset_root / rel_path
                if full_path.exists():
                    dataset_walker.append(rel_path)
                else:
                    print(f"Warning: File does not exist: {full_path}")

        print(f"Total files in dataset_walker: {len(dataset_walker)}")

        # Build walker list based on selected subset
        if subset == "training":
            exclude = set(val_list + test_list)
            self._walker = [p for p in dataset_walker if p not in exclude]
            print(f"Created training split with {len(self._walker)} files")
        elif subset == "validation":
            self._walker = val_list
            print(f"Created validation split with {len(self._walker)} files")
        else:  # "testing"
            self._walker = test_list
            print(f"Created testing split with {len(self._walker)} files")

        # Store transforms and sequence length
        self.transform = transform
        self.target_transform = target_transform
        self.sequence_length = sequence_length

        if transform:
            print("Waveform transform will be applied")
        if target_transform:
            print("Label transform will be applied")

        if len(self._walker) == 0:
            raise ValueError(f"No files found for {subset} split. Check dataset integrity or split files.")

        # Preload cache if requested
        if cache_data and preload_cache:
            print(f"Preloading {len(self._walker)} samples into memory...")
            for idx in tqdm(range(len(self)), desc="Preloading cache"):
                self._load_and_cache_sample(idx)

    def _load_split_list(self, filename: str) -> list[str]:
        txt_path = pathlib.Path(self._dataset._path) / filename
        print(f"Loading split list from {filename}")
        try:
            with txt_path.open() as f:
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
        if x.size(0) > self.sequence_length:
            return x[:self.sequence_length]
        elif x.size(0) < self.sequence_length:
            pad_size = self.sequence_length - x.size(0)
            return F.pad(x, (0, 0, 0, pad_size))
        return x

    def _load_and_cache_sample(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load a sample from disk, apply transforms, and cache it.
        """
        try:
            rel_path = self._walker[index]
            full_path = str(pathlib.Path(self._dataset._path) / rel_path)
            if not pathlib.Path(full_path).exists():
                raise FileNotFoundError(f"Audio file not found: {full_path}")
            
            waveform, _ = torchaudio.load(full_path, backend="soundfile")
            label = pathlib.Path(rel_path).parent.name

            # Apply transforms
            if self.transform:
                waveform = self.transform(waveform)  # shape: [1, n_mfcc, time]
                waveform = waveform.squeeze(0).transpose(0, 1)  # [time, n_mfcc]
                waveform = self._fix_length(waveform)
            if self.target_transform:
                label = self.target_transform(label)

            # Cache the sample if caching is enabled
            if self.cache_data:
                self.cache[index] = (waveform, label)

            return waveform, label
        except Exception as e:
            print(f"Error loading sample {index}: {str(e)}")
            raise

    def __len__(self) -> int:
        return len(self._walker)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample, using the cache if available.
        """
        if self.cache_data and index in self.cache:
            return self.cache[index]
        return self._load_and_cache_sample(index)