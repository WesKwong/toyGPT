import os
import sys

current_file_path = os.path.abspath(__file__)

parent_dir = os.path.dirname(os.path.dirname(current_file_path))

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from loguru import logger

from data.tokenizer import SimpleCharTokenizer as Tokenizer
from utils.datahelper import load_data


def test_tokenizer():
    logger.info("Running tokenizer test...")
    data_path = f"{parent_dir}/data/input.txt"
    tokenizer = Tokenizer(load_data(data_path))
    test_text = "".join(tokenizer.vocab)
    encoded_text = tokenizer.encode(test_text)
    decoded_text = tokenizer.decode(encoded_text)
    assert test_text == decoded_text, f"Expected: {list(test_text)}\n got: {list(decoded_text)}"
    logger.success("Tokenizer test passed!")

if __name__ == "__main__":
    test_tokenizer()
