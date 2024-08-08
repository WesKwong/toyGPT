from typing import List, Dict, Set

from loguru import logger
from tqdm.rich import tqdm


class SimpleCharTokenizer:
    """
    A simple character-level tokenizer.
    """

    def __init__(self, data: str) -> None:
        """
        Initialize the tokenizer.

        Args:
            char_list (List[str]): The list of characters.
        """
        self.vocab: Set[str] = set()
        self.token_to_id: Dict[str, int] = dict()
        self.id_to_token: Dict[int, str] = dict()
        self.generate_vocab(list(data))

    def generate_vocab(self, char_list: List[str]) -> None:
        """
        Generate the vocabulary from the list of characters.

        Args:
            char_list (List[str]): The list of characters.
        """
        logger.info("Tokenizer: Generating vocabulary...")
        idx = 0
        for token in tqdm(char_list):
            if token not in self.vocab:
                self.vocab.add(token)
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token
                idx += 1
        logger.info(f"Tokenizer: Vocabulary size: {len(self.vocab)}")

    def encode(self, text: str) -> List[int]:
        """
        Encode a text into a list of token ids.

        Args:
            text (str): The input text.

        Returns:
            List[int]: The list of token ids.
        """
        return [self.token_to_id[token] for token in text]

    def decode(self, tokens: List[int]) -> str:
        """
        Decode a list of token ids into a text.

        Args:
            tokens (List[int]): The list of token ids.

        Returns:
            str: The decoded text.
        """
        return "".join([self.id_to_token[token] for token in tokens])

    def __len__(self) -> int:
        return len(self.vocab)

    def __str__(self) -> str:
        return f"<SimpleCharTokenizer: vocab_size={len(self)}>"

    def __repr__(self) -> str:
        return str(self)

