import argparse
import json
import os

import torch
import yaml
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from model import EnriCo
from modules.run_evaluation import get_for_all_path, sample_train_data
from save_load import save_model, load_model


# train function
def train(model, optimizer, train_data, num_steps=1000, eval_every=100, log_dir="logs", warmup_ratio=0.1,
          train_batch_size=8, device='cuda'):
    model.train()

    # initialize data loaders
    train_loader = model.create_dataloader(train_data, batch_size=train_batch_size, shuffle=True)

    pbar = tqdm(range(num_steps))

    if warmup_ratio < 1:
        num_warmup_steps = int(num_steps * warmup_ratio)
    else:
        num_warmup_steps = int(warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_steps
    )

    iter_train_loader = iter(train_loader)

    for step in pbar:
        optimizer.zero_grad()  # Reset gradients
        try:
            x = next(iter_train_loader)
        except StopIteration:
            iter_train_loader = iter(train_loader)
            x = next(iter_train_loader)

        if step % 5 == 0:
            torch.cuda.empty_cache()

        for k, v in x.items():
            if isinstance(v, torch.Tensor):
                x[k] = v.to(device)

        try:
            loss = model(x)  # Forward pass
        except:
            continue

        # check if loss is nan
        if torch.isnan(loss):
            continue

        loss.backward()  # Compute gradients
        optimizer.step()  # Update parameters
        scheduler.step()  # Update learning rate schedule
        
        description = f"step: {step} | epoch: {step // len(train_loader)} | loss: {loss.item():.2f}"

        if (step + 1) % eval_every == 0:
            current_path = os.path.join(log_dir, f'model_{step + 1}')
            save_model(model, current_path)

            data_paths = "/gpfswork/rech/bwq/upa43yu/ie_data/RE"
            get_for_all_path(model, step, log_dir, data_paths)

            model.train()

        pbar.set_description(description)


def create_parser():
    parser = argparse.ArgumentParser(description="Span-based NER")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument('--log_dir', type=str, default='logs', help='Path to the log directory')
    return parser


def load_config_as_namespace(config_file):
    with open(config_file, 'r') as f:
        config_dict = yaml.safe_load(f)
    return argparse.Namespace(**config_dict)


if __name__ == "__main__":
    # parse args
    parser = create_parser()
    args = parser.parse_args()

    # load config
    config = load_config_as_namespace(args.config)

    config.log_dir = args.log_dir

    try:
        with open(config.train_data, 'r') as f:
            data = json.load(f)
    except:
        data = sample_train_data(config.train_data, int(config.size_sup))

    if config.prev_path != "none":
        model = load_model(config.prev_path)
        model.config = config
    else:
        model = EnriCo(config)

    if torch.cuda.is_available():
        model = model.cuda()

    lr_encoder = float(config.lr_encoder)
    lr_others = float(config.lr_others)

    optimizer = model.get_optimizer(lr_encoder, lr_others, freeze_token_rep=config.freeze_token_rep)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train(model, optimizer, data, num_steps=config.num_steps, eval_every=config.eval_every,
          log_dir=config.log_dir, warmup_ratio=config.warmup_ratio, train_batch_size=config.train_batch_size,
          device=device)
