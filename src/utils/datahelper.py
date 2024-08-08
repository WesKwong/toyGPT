def load_data(file_path: str) -> str:
    """
    Load data from a file.

    Args:
        file_path (str): The file path.

    Returns:
        str: The data.
    """
    with open(file_path, "r") as f:
        text = f.read()
    return text