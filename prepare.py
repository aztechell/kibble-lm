"""Prepare deduplicated, tokenized, story-bounded examples from TinyStories."""

import argparse
import hashlib
import json
import string
import unicodedata
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer, models, trainers

SHARDS = {
    "train": "train-00000-of-00004-2d5a1467fff1081b.parquet",
    "validation": "validation-00000-of-00001-869c898b519ad725.parquet",
}
ASCII = " " + string.ascii_letters + string.digits + string.punctuation
SPECIAL = ["<PAD>", "<BOS>", "<EOS>"]
VOCAB = 2048
MAX_PIECE = 16
TRANSLATION = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"',
                             "—": "-", "–": "-", "…": "..."})


def normalize(text):
    text = unicodedata.normalize("NFKD", text.translate(TRANSLATION))
    text = text.encode("ascii", errors="ignore").decode("ascii")
    return " ".join("".join(ch if 32 <= ord(ch) <= 126 else " " for ch in text).split())


def shard(root, split, revision):
    path = root / f"{split}.parquet"
    if not path.exists():
        root.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/datasets/roneneldan/TinyStories/resolve/{revision}/data/{SHARDS[split]}"
        temporary = path.with_suffix(".part")
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(path)
    return path


def texts(path, count, seen):
    result = []
    for batch in pq.ParquetFile(path).iter_batches(columns=["text"], batch_size=4096):
        for value in batch.column(0).to_pylist():
            text = normalize(value or "")
            digest = hashlib.sha256(text.encode("ascii")).digest()
            if text and digest not in seen:
                seen.add(digest)
                result.append(text)
                if len(result) == count:
                    return result
    raise ValueError(f"{path} has only {len(result)} eligible stories")


def encode_split(root, split, tokenizer, stories):
    # The global target index maps to a story by searchsorted, so windows never
    # include tokens from a neighboring story and need not be materialized.
    offsets = [0]
    chunks = []
    for start in range(0, len(stories), 1024):
        for story in tokenizer.encode_batch_fast(stories[start:start + 1024]):
            tokens = np.asarray([1, *story.ids, 2], dtype=np.uint16)
            chunks.append(tokens)
            offsets.append(offsets[-1] + len(tokens))
    np.save(root / f"{split}-tokens.npy", np.concatenate(chunks))
    np.save(root / f"{split}-offsets.npy", np.asarray(offsets, dtype=np.int32))
    print(f"{split}: {len(stories)} stories, {offsets[-1]} tokens", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    seen = set()
    revision = config["dataset_revision"]
    validation = texts(shard(args.data, "validation", revision), config["validation_stories"], seen)
    train = texts(shard(args.data, "train", revision), config["train_stories"], seen)

    tokenizer = Tokenizer(models.BPE())
    tokenizer.train_from_iterator(train, trainers.BpeTrainer(
        vocab_size=VOCAB, initial_alphabet=list(ASCII), special_tokens=SPECIAL,
        max_token_length=MAX_PIECE,
    ), length=len(train))
    if tokenizer.get_vocab_size() != VOCAB:
        raise ValueError("BPE vocabulary did not reach its target size")
    tokenizer.encode_special_tokens = True
    tokenizer.save(str(args.data / "tokenizer.json"))
    encode_split(args.data, "train", tokenizer, train)
    encode_split(args.data, "validation", tokenizer, validation)
    (args.data / "prepared.json").write_text(json.dumps({
        "revision": revision, "train_stories": len(train), "validation_stories": len(validation),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
