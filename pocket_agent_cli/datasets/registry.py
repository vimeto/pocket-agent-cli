"""Dataset registry for managing available benchmark datasets."""

from pathlib import Path
from typing import Dict, Type, Optional, List

from .base import Dataset


class DatasetRegistry:
    """Registry for available benchmark datasets.

    This class provides a central registry for all dataset implementations,
    allowing datasets to be discovered, instantiated, and managed.

    Usage:
        # Register a dataset (typically done with decorator)
        @DatasetRegistry.register("my_dataset")
        class MyDataset(Dataset):
            ...

        # List available datasets
        datasets = DatasetRegistry.list_datasets()

        # Create a dataset instance
        dataset = DatasetRegistry.create("mbpp", data_dir)

        # Get dataset class
        dataset_cls = DatasetRegistry.get("mbpp")
    """

    _datasets: Dict[str, Type[Dataset]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a dataset class.

        Args:
            name: Unique name for the dataset

        Returns:
            Decorator function

        Example:
            @DatasetRegistry.register("my_dataset")
            class MyDataset(Dataset):
                ...
        """
        def decorator(dataset_class: Type[Dataset]) -> Type[Dataset]:
            if name in cls._datasets:
                raise ValueError(f"Dataset '{name}' is already registered")
            if not issubclass(dataset_class, Dataset):
                raise TypeError(f"Dataset class must inherit from Dataset base class")
            cls._datasets[name] = dataset_class
            return dataset_class
        return decorator

    @classmethod
    def get(cls, name: str) -> Optional[Type[Dataset]]:
        """Get a dataset class by name.

        Args:
            name: Name of the dataset

        Returns:
            Dataset class or None if not found
        """
        return cls._datasets.get(name)

    @classmethod
    def list_datasets(cls) -> Dict[str, str]:
        """List all registered datasets with descriptions.

        Returns:
            Dictionary mapping dataset names to descriptions
        """
        result = {}
        for name, dataset_cls in cls._datasets.items():
            description = getattr(dataset_cls, 'description', 'No description')
            result[name] = description
        return result

    @classmethod
    def list_names(cls) -> List[str]:
        """List names of all registered datasets.

        Returns:
            List of dataset names
        """
        return list(cls._datasets.keys())

    @classmethod
    def create(cls, name: str, data_dir: Path) -> Dataset:
        """Create and configure a dataset instance.

        Args:
            name: Name of the dataset to create
            data_dir: Directory where dataset files are stored

        Returns:
            Configured dataset instance

        Raises:
            ValueError: If dataset name is not registered
        """
        dataset_cls = cls.get(name)
        if dataset_cls is None:
            available = ", ".join(cls.list_names()) or "none"
            raise ValueError(
                f"Unknown dataset: '{name}'. Available datasets: {available}"
            )
        return dataset_cls(data_dir)

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a dataset is registered.

        Args:
            name: Name of the dataset

        Returns:
            True if dataset is registered, False otherwise
        """
        return name in cls._datasets

    @classmethod
    def unregister(cls, name: str) -> bool:
        """Unregister a dataset (mainly for testing).

        Args:
            name: Name of the dataset to unregister

        Returns:
            True if dataset was unregistered, False if not found
        """
        if name in cls._datasets:
            del cls._datasets[name]
            return True
        return False

    @classmethod
    def clear(cls) -> None:
        """Clear all registered datasets (mainly for testing)."""
        cls._datasets.clear()

    @classmethod
    def count(cls) -> int:
        """Get number of registered datasets.

        Returns:
            Number of registered datasets
        """
        return len(cls._datasets)
