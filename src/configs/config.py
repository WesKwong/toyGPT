from dataclasses import dataclass


@dataclass
class Config:
    # toyGPT parameters
    n_block: int = 6
    seq_len: int = 128
    embed_size: int = 256
    hidden_size: int = 64
    n_head: int = 4
    expansion_factor: int = 4
    dropout: float = 0.0
    # Training parameters
    epochs: int = 20
    chunk_size: int = 64
    batch_size: int = 1536
    lr: float = 1e-3

config = Config()
