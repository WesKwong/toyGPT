import argparse

from loguru import logger

from train import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('result_id',
                        type=str,
                        help='The id of the result to use for inference')
    parser.add_argument('-g', '--generate_length',
                        type=int,
                        default=100,
                        help='The number of tokens to generate')
    return parser.parse_args()


def inference(args):
    # Set up
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    result_dir = os.path.join("results", args.result_id)
    # Load tokenizer
    raw_data = load_data("data/input.txt")
    tokenizer = SimpleCharTokenizer(raw_data)
    # Load model
    config_path = os.path.join(result_dir, "config.pt")
    config = torch.load(config_path)
    model = toyGPT(config.n_block, config.seq_len, config.embed_size,
                    config.hidden_size, config.n_head, config.expansion_factor,
                    config.dropout, len(tokenizer))
    model_path = os.path.join(result_dir, "model.pth")
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    # Inference
    while True:
        print("=" * 60)
        text = input("Enter a prompt (type 'exit' to quit)\n> ")
        if text == 'exit':
            break
        print("Generating...")
        text = tokenizer.encode(text)
        text = torch.tensor(text).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model.inference(text, args.generate_length)
        output = tokenizer.decode(output.tolist()[0])
        print(f"> {output}")

def main():
    args = parse_args()
    inference(args)

if __name__ == '__main__':
    main()

