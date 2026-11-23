"""Score KibbleLM and two TinyStories checkpoints on identical text."""

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Callable

import sentencepiece as spm
from safetensors.torch import load_file
from tokenizers import Tokenizer
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from model import BOS, CONTEXT, EOS, PAD, KibbleLM
from prepare import shard, texts


SEED = 210
STORIES = 256
PROMPTS = ("Once upon a time", "The puppy found", "One sunny day")
SAMPLE_BYTES = 160
IGNORE = -1


@dataclass
class Peer:
    name: str
    model: torch.nn.Module
    encode: Callable[[str], list[int]]
    decode: Callable[[list[int]], str]
    logits: Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor]
    vocab: int
    context: int
    checkpoint: Path


def load_peers(args, device):
    sys.path.extend(str(args.peers / path) for path in (".", "MicroMixer-4"))
    from llama2_model import ModelArgs, Transformer
    from src.model_v87_final import MicroMixerV87Final, v87_final_300k
    from src.tokenizer import ByteTokenizer

    kibble = KibbleLM().to(device).eval()
    kibble.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    bpe = Tokenizer.from_file(str(args.data / "tokenizer.json"))
    bpe.encode_special_tokens = True

    def cnn_logits(inputs, _targets):
        windows = F.pad(inputs, (CONTEXT - 1, 0), value=PAD).unfold(-1, CONTEXT, 1)
        logits = torch.cat([kibble(chunk) for chunk in windows.reshape(-1, CONTEXT).split(512)])
        return logits.reshape(*inputs.shape, -1)

    llama_path = args.peers / "stories260K.pt"
    saved = torch.load(llama_path, map_location="cpu", weights_only=True)
    llama = Transformer(ModelArgs(**saved["model_args"])).to(device).eval()
    llama.load_state_dict({key.removeprefix("_orig_mod."): value for key, value in saved["model"].items()})
    sentencepiece = spm.SentencePieceProcessor(model_file=str(args.peers / "tok512.model"))
    assert (sentencepiece.bos_id(), sentencepiece.eos_id()) == (BOS, EOS)

    mixer_path = args.peers / "epoch_2.safetensors"
    mixer = MicroMixerV87Final(v87_final_300k()).to(device).eval()
    mixer.load_state_dict(load_file(mixer_path))
    byte = ByteTokenizer()
    return (
        Peer("KibbleLM FP32", kibble, lambda text: bpe.encode(text).ids,
             lambda ids: "".join(bpe.id_to_token(token) for token in ids), cnn_logits,
             bpe.get_vocab_size(), CONTEXT, args.checkpoint),
        Peer("stories260K", llama, sentencepiece.encode, sentencepiece.decode, llama,
             sentencepiece.vocab_size(), llama.params.max_seq_len, llama_path),
        Peer("MicroMixer-4-300K", mixer, lambda text: byte.encode(text)[1:-1], byte.decode,
             lambda inputs, _: mixer(inputs)[0], byte.vocab_size, mixer.config.max_seq_len, mixer_path),
    )


def select_stories(args, peers):
    config = json.loads(Path("config.json").read_text())
    candidates = texts(shard(args.data, "validation", config["dataset_revision"]),
                       config["validation_stories"], set())
    eligible = []
    for story in candidates:
        encodings = [peer.encode(story) for peer in peers]
        if any(peer.decode(ids) != story for peer, ids in zip(peers, encodings)):
            continue
        # The CNN slides its 32-token window over stories of any length.
        if any(len(ids) > peer.context for peer, ids in zip(peers[1:], encodings[1:])):
            continue
        eligible.append((story, encodings))
    if len(eligible) < STORIES:
        raise ValueError(f"Only {len(eligible)} stories fit every model and round-trip cleanly")
    return random.Random(SEED).sample(eligible, STORIES), len(eligible), config["dataset_revision"]


def padded_batch(encodings, device):
    inputs = [torch.tensor([BOS, *ids[:-1]]) for ids in encodings]
    targets = [torch.tensor(ids) for ids in encodings]
    return (pad_sequence(inputs, batch_first=True, padding_value=PAD).to(device),
            pad_sequence(targets, batch_first=True, padding_value=IGNORE).to(device))


def score(peer, encoded, byte_count, device):
    encoded.sort(key=len)
    nll = 0.0
    token_count = sum(map(len, encoded))
    for start in range(0, len(encoded), 8):
        inputs, targets = padded_batch(encoded[start:start + 8], device)
        logits = peer.logits(inputs, targets)
        nll += F.cross_entropy(logits.flatten(0, 1), targets.flatten(),
                               ignore_index=IGNORE, reduction="sum").item()
    return {"bits_per_byte": nll / (math.log(2) * byte_count), "tokens": token_count,
            "perplexity": math.exp(nll / token_count)}


def verify_causality(peer, device):
    ids = torch.tensor([[BOS, *peer.encode("One sunny day, a little fox found a red hat.")]], device=device)
    prefix = ids[:, :ids.shape[1] // 2].contiguous()
    scores = peer.logits(ids, ids)
    prefix_scores = peer.logits(prefix, prefix)
    torch.testing.assert_close(scores[:, :prefix.shape[1]], prefix_scores, atol=3e-4, rtol=3e-4)


def sample(peer, prompt, device):
    ids = [BOS, *peer.encode(prompt)]
    sampler = torch.Generator(device="cpu").manual_seed(SEED)
    continuation = b""
    for _ in range(SAMPLE_BYTES):
        inputs = torch.tensor([ids[-peer.context:]], device=device)
        logits = peer.logits(inputs, None)[0, -1].clone()
        logits[:EOS] = -torch.inf
        values, tokens = logits.topk(16)
        index = torch.multinomial((values.cpu() / 0.85).softmax(0), 1, generator=sampler)
        token = int(tokens[index].item())
        if token == EOS:
            break
        ids.append(token)
        continuation = peer.decode(ids[1:])[len(prompt):].encode("utf-8")
        if len(continuation) >= SAMPLE_BYTES:
            break
    return continuation[:SAMPLE_BYTES].decode("utf-8", errors="ignore")


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/v2/best.pt"))
    parser.add_argument("--peers", type=Path, default=Path("runs/peer-comparison"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(4)
    peers = load_peers(args, device)
    selected, eligible, revision = select_stories(args, peers)
    normalized = [story for story, _ in selected]
    byte_count = sum(len(story.encode("ascii")) for story in normalized)
    result = {"dataset_revision": revision, "seed": SEED, "stories": len(selected), "eligible": eligible,
              "bytes": byte_count, "sample_sha256": hashlib.sha256("\n".join(normalized).encode("ascii")).hexdigest(),
              "pytorch": str(torch.__version__), "device": str(device), "models": {}}
    for index, peer in enumerate(peers):
        verify_causality(peer, device)
        metrics = score(peer, [encodings[index] for _, encodings in selected], byte_count, device)
        metrics.update(params=sum(p.numel() for p in peer.model.parameters()),
                       vocab=peer.vocab, context=peer.context,
                       checkpoint_bytes=peer.checkpoint.stat().st_size,
                       checkpoint_sha256=hashlib.sha256(peer.checkpoint.read_bytes()).hexdigest(),
                       samples={prompt: sample(peer, prompt, device) for prompt in PROMPTS})
        result["models"][peer.name] = metrics
        print(f"{peer.name}: {metrics['params']:,} params, {metrics['bits_per_byte']:.4f} bits/byte", flush=True)
    path = args.out or args.peers / "comparison.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
