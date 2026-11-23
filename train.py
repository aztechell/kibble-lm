"""Train KibbleLM and evaluate the best checkpoint on the frozen validation set."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model import BOS, CONTEXT, EOS, PAD, PROMPTS, VOCAB, KibbleLM


class Corpus:
    def __init__(self, directory, split, device):
        self.tokens = torch.from_numpy(np.load(directory / f"{split}-tokens.npy").astype(np.int32)).to(device)
        self.offsets = torch.from_numpy(np.load(directory / f"{split}-offsets.npy")).to(device)
        # Remove BOS positions from the sampling index, one per story.
        self.targets = self.offsets - torch.arange(len(self.offsets), device=device, dtype=torch.int32)
        self.lookback = torch.arange(CONTEXT, 0, -1, device=device)

    def sample(self, count, generator):
        return torch.randint(len(self.tokens) - len(self.offsets) + 1, (count,),
                             device=self.tokens.device, generator=generator)

    def batch(self, indices):
        stories = torch.searchsorted(self.targets[1:], indices, right=True)
        positions = indices + stories + 1
        start = self.offsets[stories, None]
        context = positions[:, None] - self.lookback
        words = self.tokens[context.maximum(start)]
        return words.masked_fill(context < start, PAD), self.tokens[positions].long()

    def bigrams(self):
        pairs = self.tokens[:-1].long() * VOCAB + self.tokens[1:]
        pairs = pairs[self.tokens[1:] != BOS]
        counts = torch.bincount(pairs, minlength=VOCAB * VOCAB).reshape(VOCAB, VOCAB)
        counts = counts.to(torch.float32) + 0.1
        return torch.log(counts / counts.sum(1, keepdim=True))


@torch.no_grad()
def evaluate(model, corpus, indices, batch_size, bigrams=None):
    model.eval()
    loss = torch.zeros((), device=indices.device)
    baseline_loss = torch.zeros_like(loss)
    for batch in indices.split(batch_size):
        context, targets = corpus.batch(batch)
        loss += F.cross_entropy(model(context), targets, reduction="sum")
        if bigrams is not None:
            baseline_loss -= bigrams[context[:, -1].long(), targets].sum()
    return float(loss / len(indices)), float(baseline_loss / len(indices))


@torch.no_grad()
def generate(model, tokenizer, prompt, device, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    tokens = [BOS, *tokenizer.encode(prompt).ids]
    output = []
    for _ in range(48):
        context = ([PAD] * CONTEXT + tokens)[-CONTEXT:]
        logits = model(torch.tensor([context], device=device))[0]
        logits[:EOS] = -torch.inf
        values, indices = logits.topk(16)
        sample = torch.multinomial((values / 0.85).softmax(0), 1, generator=generator)
        token = int(indices[sample].item())
        if token == EOS:
            break
        tokens.append(token)
        output.append(tokenizer.id_to_token(token))
    return prompt + "".join(output)


def atomic_save(path, model):
    temporary = path.with_suffix(".tmp")
    torch.save(model.state_dict(), temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("runs/v2"))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.steps is not None:
        config["train_steps"] = args.steps
    args.out.mkdir(parents=True, exist_ok=True)
    config["device"] = args.device
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    seed = config["seed"]
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch is required for the default training profile")
    train = Corpus(args.data, "train", device)
    validation = Corpus(args.data, "validation", device)
    train_rng = torch.Generator(device=device).manual_seed(seed + 1)
    validation_rng = torch.Generator(device=device).manual_seed(seed + 2)
    validation_indices = validation.sample(config["validation_tokens"], validation_rng)
    np.save(args.out / "validation-indices.npy", validation_indices.cpu().numpy())

    model = KibbleLM().to(device=device, memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  fused=device.type == "cuda")
    steps = config["train_steps"]
    batch_size = config["batch_size"]
    path = args.out / "best.pt"
    best = math.inf
    history = []
    print(f"Training {steps} steps on {device}", flush=True)
    for step in range(1, steps + 1):
        model.train()
        context, targets = train.batch(train.sample(batch_size, train_rng))
        progress = step / steps
        multiplier = min(step / 200, 1.0) * (0.15 + 0.85 * (1 + math.cos(math.pi * progress)) / 2)
        optimizer.param_groups[0]["lr"] = config["learning_rate"] * multiplier
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss = F.cross_entropy(model(context), targets)
        loss.backward()
        optimizer.step()

        if step % config["eval_interval"] == 0 or step == steps:
            val_loss, _ = evaluate(model, validation, validation_indices, batch_size)
            entry = {"step": step, "train_loss": float(loss.detach()), "validation_loss": val_loss}
            history.append(entry)
            print(json.dumps(entry), flush=True)
            if val_loss < best:
                best = val_loss
                atomic_save(path, model)

    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    val_loss, baseline = evaluate(model, validation, validation_indices, batch_size, train.bigrams())
    tokenizer = Tokenizer.from_file(str(args.data / "tokenizer.json"))
    tokenizer.encode_special_tokens = True
    report = {
        "train_steps": steps,
        "validation_perplexity": math.exp(val_loss),
        "bigram_perplexity": math.exp(baseline),
        "history": history,
        "samples": [generate(model, tokenizer, prompt, device, seed) for prompt in PROMPTS],
    }
    (args.out / "training.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
