import torch
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
from typing import Optional

class SMNISTDataModule:
    """
    A data module for loading and processing the Sequential MNIST (sMNIST) dataset,
    with optional support for the permuted sMNIST variant.
    """
    def __init__(
        self,
        data_dir: str = "data",
        batch_size: int = 256,
        val_split: float = 0.1,
        permuted: bool = True,
        seed: Optional[int] = None,
        pin_memory: Optional[bool] = None,
        num_workers: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initializes the SMNIST data module.

        Args:
            data_dir (str): Directory to download/load the MNIST dataset.
            batch_size (int): Batch size used for all loaders.
            val_split (float): Proportion of training data to use for validation.
            permuted (bool): Whether to apply a fixed random permutation to the sequence.
            seed (Optional[int]): Optional seed for reproducibility.
            pin_memory (Optional[bool]): If True, pin memory during data loading (recommended for CUDA).
            num_workers (Optional[int]): Number of subprocesses for data loading.
            device (Optional[torch.device]): Torch device to use for tensor operations.
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_split = val_split
        self.permuted = permuted
        self.sequence_length = 28 * 28  # 784 for flattened MNIST images
        self.input_size = 1
        self.num_classes = 10

        # Set device and data loader behavior
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pin_memory = pin_memory if pin_memory is not None else (self.device.type == "cuda")
        self.num_workers = num_workers if num_workers is not None else (1 if self.device.type == "cuda" else 0)

        if seed is not None:
            torch.manual_seed(seed)

        # Generate or load permutation index
        self.permuted_idx = self._init_permutation(seed)

        # Create the train/val/test loaders
        self.train_loader, self.val_loader, self.test_loader = self._load_data()

    def _init_permutation(self, seed: Optional[int]):
        """
        Creates a random permutation index for permuted sMNIST if needed.

        Args:
            seed (Optional[int]): Optional random seed to control reproducibility.

        Returns:
            torch.Tensor: The permutation index.
        """
        if self.permuted:
            permuted_idx = torch.randperm(self.sequence_length)
            perm_save_path = Path("models")
            perm_save_path.mkdir(exist_ok=True)

            # Save the permutation for reproducibility and debugging
            rand_str = str(seed or torch.randint(1, 9999, (1,)).item())
            perm_file = perm_save_path / f"{rand_str}_permuted_idx.pt"
            torch.save(permuted_idx, perm_file)
            return permuted_idx
        else:
            return torch.arange(self.sequence_length)

    def _load_data(self):
        """
        Downloads and prepares the MNIST dataset, splits it into train/val/test loaders.

        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]: The train, validation, and test loaders.
        """
        transform = transforms.ToTensor()

        # Load full training set
        train_full = torchvision.datasets.MNIST(
            root=self.data_dir, train=True, transform=transform, download=True
        )

        # Create train/val split
        val_size = int(len(train_full) * self.val_split)
        train_size = len(train_full) - val_size
        train_set, val_set = random_split(train_full, [train_size, val_size])

        # Load test set
        test_set = torchvision.datasets.MNIST(
            root=self.data_dir, train=False, transform=transform, download=True
        )

        # DataLoaders
        train_loader = DataLoader(
            dataset=train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        val_loader = DataLoader(
            dataset=val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        test_loader = DataLoader(
            dataset=test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

        return train_loader, val_loader, test_loader

    def transform_batch(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Reshapes and permutes a batch of inputs into sequential form.

        Args:
            inputs (torch.Tensor): Input tensor of shape [B, 1, 28, 28].

        Returns:
            torch.Tensor: Transformed tensor of shape [T, B, D], where T = sequence length.
        """
        batch_size = inputs.size(0)

        # Flatten and reshape to [B, 784, 1]
        x = inputs.to(device=self.device).view(batch_size, self.sequence_length, self.input_size)

        # Permute to [T, B, D]
        x = x.permute(1, 0, 2)

        # Apply permutation across the time dimension
        return x[self.permuted_idx]

    def get_loaders(self):
        """
        Returns the train, validation, and test data loaders.

        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]
        """
        return self.train_loader, self.val_loader, self.test_loader

    def get_permutation(self):
        """
        Returns the permutation index used for permuted sMNIST.

        Returns:
            torch.Tensor
        """
        return self.permuted_idx

if __name__ == "__main__":
    import random

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the data module
    smnist = SMNISTDataModule(
        data_dir="data",
        batch_size=128,
        permuted=True,
        seed=random.randint(1, 9999),
        device=device
    )

    # Get the data loaders
    train_loader, val_loader, test_loader = smnist.get_loaders()

    # Fetch a batch
    for inputs, targets in train_loader:
        # Transform batch for time-series model input
        inputs_seq = smnist.transform_batch(inputs)
        print(f"Input shape (T, B, D): {inputs_seq.shape}")
        print(f"Targets shape: {targets.shape}")
        break
