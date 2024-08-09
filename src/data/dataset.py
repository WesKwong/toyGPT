from typing import Tuple, Any
from typing_extensions import TypeAlias

import numpy as np
import numpy.typing as npt
from numba import njit, prange
from loguru import logger
from torch.utils.data import Dataset, DataLoader

DataType: TypeAlias = npt.NDArray[np.longlong]


@njit(parallel=True)
def gen_dataset(encoded_data: DataType, len: int,
                chunk_size: int) -> Tuple[DataType, DataType]:
    """
    Generate datas and labels from the encoded data.

    Args:
        encoded_data (DataType): The encoded data.
        len (int): The length of the encoded data.
        chunk_size (int): The size of the chunk.

    Returns:
        The datas and labels are both of shape `(len, chunk_size)` -> `Tuple[DataType, DataType]`
    """
    datas = np.empty((len, chunk_size), dtype=np.longlong)
    labels = np.empty((len, chunk_size), dtype=np.longlong)
    for i in prange(len):
        datas[i] = encoded_data[i:i + chunk_size]
        labels[i] = encoded_data[i + 1:i + chunk_size + 1]
    return datas, labels


class ShakespeareDataset(Dataset):
    """
    A PyTorch dataset class for the Shakespeare dataset.
    """

    def __init__(self, raw_data: str, tokenizer: Any, chunk_size: int) -> None:
        """
        Initialize the dataset.

        Args:
            datapath (str): The path to the dataset.
            tokenizer (Any): The tokenizer.
        """
        self.len: int = 0
        self.datas: np.NDArray[np.longlong]
        self.labels: np.NDArray[np.longlong]

        encoded_data = tokenizer.encode(raw_data)
        self.len: int = len(encoded_data) - chunk_size
        logger.info(f"Dataset: Generating datas and labels...")
        self.datas, self.labels = gen_dataset(encoded_data, self.len,
                                              chunk_size)
        logger.info(f"Dataset: Length: {self.len}")

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return self.len

    def __getitem__(self, idx: int) -> Tuple[DataType, DataType]:
        """
        Get an item from the dataset.

        Args:
            idx (int): The index of the item.

        Returns:
            The datas and labels -> `Tuple[DataType, DataType]`
        """
        return self.datas[idx], self.labels[idx]


def get_dataloader(dataset: ShakespeareDataset, batch_size: int, train: bool,
                   num_workers: int = 4) -> DataLoader:
    """
    Get a dataloader for the dataset.

    Args:
        dataset (ShakespeareDataset): The dataset.
        batch_size (int): The batch size.
        train (bool): Whether the dataloader is for training.
        num_workers (int): The number of workers.

    Returns:
        DataLoader: The dataloader.
    """
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=train,
                      num_workers=num_workers)
