"""Train KibbleLM and evaluate the best checkpoint on the frozen validation set."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model import BOS, CONTEXT, EOS, FEATURES, INNER_CHANNELS, PAD, PROMPTS, VOCAB, KibbleLM
from qat import KpuQat
from teacher import TeacherLM


DISTILL_BLOCK = 16
DISTILL_TEMPERATURE = 1.5


class Corpus:
    def __init__(self, directory, split, device, context=CONTEXT):
        tokens = np.load(directory / f"{split}-tokens.npy", mmap_mode="c")
        self.tokens = torch.from_numpy(tokens).to(device=device, dtype=torch.int32)
        self.offsets = torch.from_numpy(np.load(directory / f"{split}-offsets.npy")).to(device)
        # Remove BOS positions from the sampling index, one per story.
        self.targets = self.offsets - torch.arange(len(self.offsets), device=device, dtype=torch.int32)
        self.lookback = torch.arange(context, 0, -1, device=device)

    def sample(self, count, generator):
        return torch.randint(len(self.tokens) - len(self.offsets) + 1, (count,),
                             device=self.tokens.device, generator=generator)

    def story_bounds(self, indices):
        stories = torch.searchsorted(self.targets[1:], indices, right=True)
        return self.offsets[stories, None], indices + stories + 1

    def batch(self, indices):
        start, positions = self.story_bounds(indices)
        context = positions[:, None] - self.lookback
        words = self.tokens[context.maximum(start)]
        return words.masked_fill(context < start, PAD), self.tokens[positions].long()

    def sequences(self, indices, length):
        """Return contiguous teacher inputs and next-token labels within one story."""
        start, end = self.story_bounds(indices)
        positions = end[:, None] - torch.arange(length - 1, -1, -1, device=end.device)
        input_positions = positions - 1
        tokens = self.tokens[input_positions.maximum(start)].masked_fill(input_positions < start, PAD)
        labels = self.tokens[positions.maximum(start)].long().masked_fill(positions <= start, -100)
        return tokens, labels

    def bigrams(self):
        counts = torch.zeros(VOCAB * VOCAB, device=self.tokens.device, dtype=torch.int64)
        for previous, target in zip(self.tokens[:-1].split(1 << 20), self.tokens[1:].split(1 << 20)):
            pairs = previous.long() * VOCAB + target
            counts += torch.bincount(pairs[target != BOS], minlength=counts.numel())
        counts = counts.reshape(VOCAB, VOCAB).to(torch.float32) + 0.1
        return torch.log(counts / counts.sum(1, keepdim=True))


@torch.no_grad()
def evaluate(model, corpus, indices, batch_size, bigrams=None, forward=None):
    model.eval()
    forward = forward or model
    loss = torch.zeros((), device=indices.device)
    baseline_loss = torch.zeros_like(loss)
    for batch in indices.split(batch_size):
        context, targets = corpus.batch(batch)
        loss += F.cross_entropy(forward(context), targets, reduction="sum")
        if bigrams is not None:
            baseline_loss -= bigrams[context[:, -1].long(), targets].sum()
    return float(loss / len(indices)), float(baseline_loss / len(indices))


@torch.no_grad()
def generate(model, tokenizer, prompt, device, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    tokens = [BOS, *tokenizer.encode(prompt).ids]
    output = []
    for _ in range(48):
        context = ([PAD] * model.context + tokens)[-model.context:]
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


def atomic_save(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def initialize(model, source):
    state = model.state_dict()
    if state.keys() != source.keys():
        raise ValueError("Initialization checkpoint has a different layer layout")
    for name, tensor in state.items():
        saved = source[name]
        if saved.shape == tensor.shape:
            tensor.copy_(saved)
            continue
        if saved.ndim != tensor.ndim or any(old > new for old, new in zip(saved.shape, tensor.shape)):
            raise ValueError(f"Cannot initialize {name} from {tuple(saved.shape)} into {tuple(tensor.shape)}")
        if tensor.ndim >= 2:
            # Existing output channels initially ignore the added input channels.
            tensor[:saved.shape[0]].zero_()
        tensor[tuple(slice(0, size) for size in saved.shape)].copy_(saved)


def distillation_loss(model, teacher, corpus, indices):
    tokens, labels = corpus.sequences(indices, teacher.context)
    labels = labels[:, -DISTILL_BLOCK:]
    windows = tokens.unfold(-1, model.context, 1)[:, -DISTILL_BLOCK:]
    logits = model(windows.flatten(0, 1)).float()
    with torch.no_grad():
        teacher_logits = teacher.logits(tokens, DISTILL_BLOCK).flatten(0, 1).float()
        target_probabilities = (teacher_logits / DISTILL_TEMPERATURE).softmax(-1)
    valid = labels.flatten() != -100
    cross_entropy = F.cross_entropy(logits, labels.flatten())
    log_probabilities = (logits / DISTILL_TEMPERATURE).log_softmax(-1)
    soft_loss = -(target_probabilities * log_probabilities).sum(-1)
    return 0.65 * cross_entropy + 0.35 * DISTILL_TEMPERATURE ** 2 * soft_loss[valid].mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("runs/v2"))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--context", type=int, choices=(32, 64))
    parser.add_argument("--inner-channels", type=int, nargs=3)
    parser.add_argument("--features", type=int, choices=(64, 72, 76))
    parser.add_argument("--layout", choices=("snake", "spiral"))
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--train-seed", type=int)
    parser.add_argument("--init-checkpoint", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train-teacher", action="store_true")
    mode.add_argument("--teacher-checkpoint", type=Path)
    mode.add_argument("--qat", action="store_true", help="Fine-tune for the K210 quantized path")
    parser.add_argument("--resume", action="store_true", help="Continue the same run from last.pt")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.steps is not None:
        config["train_steps"] = args.steps
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    config["context"] = args.context or config.get("context", CONTEXT)
    config["inner_channels"] = args.inner_channels or config.get("inner_channels", list(INNER_CHANNELS))
    if args.features:
        config["features"] = args.features
    if args.layout:
        config["layout"] = args.layout
    if args.train_seed is not None:
        config["train_seed"] = args.train_seed
    if args.learning_rate is not None:
        config["learning_rate"] = args.learning_rate
    if args.init_checkpoint:
        config["init_checkpoint"] = str(args.init_checkpoint)
    if args.train_teacher:
        config.update(context=TeacherLM.context, model_type="teacher")
    if args.qat:
        if not args.init_checkpoint:
            parser.error("--qat requires --init-checkpoint")
        config["qat"] = True
    if args.teacher_checkpoint:
        config["teacher_checkpoint"] = str(args.teacher_checkpoint)
        if config["batch_size"] % DISTILL_BLOCK:
            parser.error(f"Distillation batch size must be divisible by {DISTILL_BLOCK}")
    args.out.mkdir(parents=True, exist_ok=True)
    config["device"] = args.device
    config_path = args.out / "config.json"
    if args.resume:
        if config != json.loads(config_path.read_text()):
            raise ValueError("Resume requires the same training configuration")
    else:
        config_path.write_text(json.dumps(config, indent=2) + "\n")

    seed = config["seed"]
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch is required for the default training profile")
    train = Corpus(args.data, "train", device, config["context"])
    validation = Corpus(args.data, "validation", device, config["context"])
    train_rng = torch.Generator(device=device).manual_seed(config.get("train_seed", seed + 1))
    validation_rng = torch.Generator(device=device).manual_seed(seed + 2)
    validation_indices = validation.sample(config["validation_tokens"], validation_rng)
    np.save(args.out / "validation-indices.npy", validation_indices.cpu().numpy())

    if args.train_teacher:
        model = TeacherLM().to(device=device)
    else:
        model = KibbleLM(config["context"], config["inner_channels"], config.get("layout", "snake"),
                         config.get("features", FEATURES)).to(device=device, memory_format=torch.channels_last)
    teacher = None
    if args.teacher_checkpoint:
        teacher = TeacherLM().to(device=device).eval()
        teacher.load_state_dict(torch.load(args.teacher_checkpoint, map_location="cpu", weights_only=True))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  fused=device.type == "cuda")
    steps = config["train_steps"]
    batch_size = config["batch_size"]
    path = args.out / "best.pt"
    last_path = args.out / "last.pt"
    start = 1
    best = math.inf
    history = []
    if args.resume:
        state = torch.load(last_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        train_rng.set_state(state["train_rng"])
        start, best, history = state["step"] + 1, state["best"], state["history"]
        del state
    elif args.init_checkpoint:
        initialize(model, torch.load(args.init_checkpoint, map_location="cpu", weights_only=True))
        best, _ = evaluate(model, validation, validation_indices, batch_size)
        entry = {"step": 0, "validation_loss": best}
        history.append(entry)
        atomic_save(path, model.state_dict())
        print(json.dumps(entry), flush=True)
    qat = KpuQat(model, train) if args.qat else None
    if qat is not None and not args.resume:
        best, _ = evaluate(model, validation, validation_indices, batch_size, forward=qat)
        history[-1]["simulated_quantized_loss"] = best
        print(json.dumps(history[-1]), flush=True)
    print(f"Training steps {start} to {steps} on {device}, context {model.context}", flush=True)
    for step in range(start, steps + 1):
        model.train(qat is None)
        progress = step / steps
        multiplier = min(step / 200, 1.0) * (0.15 + 0.85 * (1 + math.cos(math.pi * progress)) / 2)
        optimizer.param_groups[0]["lr"] = config["learning_rate"] * multiplier
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            if teacher is not None:
                indices = train.sample(batch_size // DISTILL_BLOCK, train_rng)
                loss = distillation_loss(model, teacher, train, indices)
            elif args.train_teacher:
                tokens, labels = train.sequences(train.sample(batch_size, train_rng), model.context)
                logits = model.logits(tokens, model.context)
                loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
            else:
                context, targets = train.batch(train.sample(batch_size, train_rng))
                loss = F.cross_entropy(model(context), targets)
                if qat is not None:
                    loss = 0.25 * loss + 0.75 * F.cross_entropy(qat(context), targets)
        loss.backward()
        optimizer.step()
        if qat is not None and step % 250 == 0:
            qat.refresh_ranges()

        if step % config["eval_interval"] == 0 or step == steps:
            val_loss, _ = evaluate(model, validation, validation_indices, batch_size)
            entry = {"step": step, "train_loss": float(loss.detach()), "validation_loss": val_loss}
            selection_loss = val_loss
            if qat is not None:
                qat.refresh_ranges()
                selection_loss, _ = evaluate(model, validation, validation_indices, batch_size, forward=qat)
                entry["simulated_quantized_loss"] = selection_loss
            history.append(entry)
            print(json.dumps(entry), flush=True)
            if selection_loss < best:
                best = selection_loss
                atomic_save(path, model.state_dict())
            atomic_save(last_path, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                    "train_rng": train_rng.get_state(), "step": step,
                                    "best": best, "history": history})

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
    if qat is not None:
        qat.refresh_ranges()
        quantized_loss, _ = evaluate(model, validation, validation_indices, batch_size, forward=qat)
        report["simulated_quantized_perplexity"] = math.exp(quantized_loss)
    (args.out / "training.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
