import os
import sys
import pathlib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from typing import Tuple, Optional
from torch.utils.data import DataLoader
from torchaudio.transforms import MFCC
from gsc.data.SpeechCommands import SpeechCommands
from gsc.data.OneHotTargetTransform import OneHotTargetTransform

class SpeechCommandsDataLoader:
    """
    A class that sets up and returns PyTorch DataLoaders for the SpeechCommands dataset.

    Parameters:
    - root (str): Root directory for the dataset.
    - sequence_length (int): Number of time steps to pad/truncate each sample to.
    - batch_size (int): Batch size for all splits.
    - num_workers (int): Number of worker threads for loading.
    - pin_memory (bool): Whether to pin memory (for CUDA optimization).
    - cache_data (bool): Whether to cache data in memory. Defaults to True.
    - preload_cache (bool): Whether to preload all data into cache. Defaults to False.
    - data_percentage (float): Percentage of data to use (0.0 to 100.0). Defaults to 100.0.
    - seed (int, optional): Random seed for reproducible data sampling. Defaults to None.
    """
    def __init__(
        self,
        root: str,
        sequence_length: int = 1300,
        batch_size: int = 4,
        num_workers: int = 0,
        pin_memory: bool = False,
        cache_data: bool = True,
        preload_cache: bool = False,
        data_percentage: float = 100.0,
        seed: Optional[int] = None,
    ):
        self.root = root
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.cache_data = cache_data
        self.preload_cache = preload_cache
        self.data_percentage = data_percentage
        self.seed = seed

        print("Building label-to-index map...")
        base_ds = SpeechCommands(
            root=root,
            subset="training",
            download=True,
            cache_data=False,
            data_percentage=100.0
        )
        self.labels = sorted(set(pathlib.Path(path).parent.name for path in base_ds._walker))
        self.label_to_index = {label: idx for idx, label in enumerate(self.labels)}
        self.num_classes = len(self.labels)
        print(f"Detected {self.num_classes} unique labels")

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
        dataset = SpeechCommands(
            root=self.root,
            subset=subset,
            transform=self.mfcc_transform,
            target_transform=self.label_transform,
            sequence_length=self.sequence_length,
            download=False,
            cache_data=self.cache_data,
            preload_cache=self.preload_cache,
            data_percentage=self.data_percentage,
            seed=self.seed
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
        print("Creating train/val/test loaders...")
        train_loader = self._create_loader("training")
        val_loader = self._create_loader("validation")
        test_loader = self._create_loader("testing")
        print("All loaders created successfully.")
        return train_loader, val_loader, test_loader