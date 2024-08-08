import os
import sys

current_file_path = os.path.abspath(__file__)

parent_dir = os.path.dirname(os.path.dirname(current_file_path))

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from loguru import logger
import numpy as np

from data.tokenizer import SimpleCharTokenizer as Tokenizer
from data.dataset import ShakespeareDataset as Dataset
from utils.datahelper import load_data


def test_dataset():
    logger.info("Running dataset test...")
    data_path = f"{parent_dir}/data/input.txt"
    raw_data = load_data(data_path)
    tokenizer = Tokenizer(raw_data)
    dataset = Dataset(raw_data, tokenizer, 100)
    encoded_data = np.array(tokenizer.encode(raw_data), dtype=np.int32)
    assert len(dataset) == len(encoded_data) - 100
    assert len(dataset[0][0]) == 100
    def check_same(a, b):
        return all(a == b)
    assert check_same(dataset[0][0], encoded_data[:100]), \
        f"Expected: {encoded_data[:100]}\n got: {dataset[0][0]}"
    assert check_same(dataset[0][1], encoded_data[1:101]), \
        f"Expected: {encoded_data[1:101]}\n got: {dataset[0][1]}"
    assert check_same(dataset[-1][0], encoded_data[-101:-1]), \
        f"Expected: {encoded_data[-101:-1]}\n got: {dataset[-1][0]}"
    assert check_same(dataset[-1][1], encoded_data[-100:]), \
        f"Expected: {encoded_data[-100:]}\n got: {dataset[-1][1]}"
    logger.success("Dataset test passed!")

if __name__ == "__main__":
    test_dataset()
