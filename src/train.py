import os
import sys
import random

import torch
from tqdm.rich import tqdm
from loguru import logger
import matplotlib.pyplot as plt

from model import toyGPT
from data import ShakespeareDataset, SimpleCharTokenizer, get_dataloader
from utils import load_data
from configs import config

def make_result_dir():
    exp_id = random.randint(100000, 999999)
    result_dir = os.path.join("results", str(exp_id))
    os.makedirs(result_dir, exist_ok=True)
    return result_dir

def train_model(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for inputs, targets in tqdm(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        logits, loss = model(inputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if total_loss != total_loss:
            raise ValueError("NaN detected!")
    return total_loss / len(train_loader)

@torch.no_grad()
def valid_model(model, valid_loader, device):
    model.eval()
    total_loss = 0
    for inputs, targets in tqdm(valid_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        logits, loss = model(inputs, targets)
        total_loss += loss.item()
        if total_loss != total_loss:
            raise ValueError("NaN detected!")
    return total_loss / len(valid_loader)

def plot(train_losses, valid_losses, result_dir):
    epochs = len(train_losses)
    plt.figure()
    plt.plot(range(epochs), train_losses, label='Train')
    if valid_losses is not None:
        plt.plot(range(epochs), valid_losses, label='Valid')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Valid Loss')
    save_path = os.path.join(result_dir, f"loss-{epochs}.png")
    plt.savefig(save_path)

def train(model, train_loader, valid_loader, optimizer, epochs, device, result_dir):
    train_loss_history = []
    valid_loss_history = []
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        # Train
        logger.info("Training...")
        train_loss = train_model(model, train_loader, optimizer, device)
        train_loss_history.append(train_loss)
        logger.info(f"Train Loss: {train_loss}")
        # Validation
        logger.info("Validating...")
        valid_loss = valid_model(model, valid_loader, device)
        valid_loss_history.append(valid_loss)
        logger.info(f"Valid Loss: {valid_loss}")
        # plot
        plot(train_loss_history, valid_loss_history, result_dir)
    loss_history = dict(train=train_loss_history, valid=valid_loss_history)
    save_path = os.path.join(result_dir, "loss_history.pt")
    torch.save(loss_history, save_path)
    save_path = os.path.join(result_dir, f"model.pth")
    torch.save(model.state_dict(), save_path)


def main():
    # Set up
    result_dir = make_result_dir()
    torch.save(config, os.path.join(result_dir, "config.pt"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    # Load data
    raw_data = load_data("data/input.txt")
    tokenizer = SimpleCharTokenizer(raw_data)
    dataset = ShakespeareDataset(raw_data, tokenizer, config.chunk_size)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [int(len(dataset) * 0.8),
         len(dataset) - int(len(dataset) * 0.8)])
    train_loader = get_dataloader(train_dataset, config.batch_size, True)
    valid_loader = get_dataloader(val_dataset, config.batch_size, False)
    # Load model
    model = toyGPT(config.n_block, config.seq_len, config.embed_size,
                    config.hidden_size, config.n_head, config.expansion_factor,
                    config.dropout, len(tokenizer))
    if sys.version_info < (3, 12):
        model = torch.compile(model, mode='reduce_overhead')
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    train(model, train_loader, valid_loader, optimizer, config.epochs, device, result_dir)

if __name__ == "__main__":
    main()
