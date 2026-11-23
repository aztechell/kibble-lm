"""Prepare deduplicated, tokenized, story-bounded examples from TinyStories."""

import argparse
import hashlib
from itertools import islice
import json
import string
import tempfile
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
TRAIN_SHARDS = (
    SHARDS["train"],
    "train-00001-of-00004-5852b56a2bd28fd9.parquet",
    "train-00002-of-00004-a26307300439e943.parquet",
    "train-00003-of-00004-d243063613e5a057.parquet",
)
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


def shard(root, split, revision, index=0):
    name = TRAIN_SHARDS[index] if split == "train" else SHARDS[split]
    path = root / (name if index else f"{split}.parquet")
    if not path.exists():
        root.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/datasets/roneneldan/TinyStories/resolve/{revision}/data/{name}"
        temporary = path.with_suffix(".part")
        print(f"Downloading {name}", flush=True)
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(path)
    return path


def unique_texts(path, seen):
    for batch in pq.ParquetFile(path).iter_batches(columns=["text"], batch_size=4096):
        for value in batch.column(0).to_pylist():
            text = normalize(value or "")
            digest = hashlib.sha256(text.encode("ascii")).digest()
            if text and digest not in seen:
                seen.add(digest)
                yield text


def texts(path, count, seen):
    result = list(islice(unique_texts(path, seen), count))
    if len(result) != count:
        raise ValueError(f"{path} has only {len(result)} eligible stories")
    return result


def training_texts(cache, revision, seen):
    for index in range(len(TRAIN_SHARDS)):
        yield from unique_texts(shard(cache, "train", revision, index), seen)


def encode_split(root, split, tokenizer, stories):
    # The global target index maps to a story by searchsorted, so windows never
    # include tokens from a neighboring story and need not be materialized.
    offsets = [0]
    with tempfile.TemporaryFile(dir=root) as binary:
        stories = iter(stories)
        while batch := list(islice(stories, 1024)):
            chunks = []
            for story in tokenizer.encode_batch_fast(batch):
                tokens = np.asarray([1, *story.ids, 2], dtype=np.uint16)
                chunks.append(tokens)
                offsets.append(offsets[-1] + len(tokens))
            np.concatenate(chunks).tofile(binary)
            if (len(offsets) - 1) % (128 * 1024) == 0:
                print(f"{split}: encoded {len(offsets) - 1:,} stories", flush=True)
        if offsets[-1] == 0:
            raise ValueError(f"Empty {split} split")
        binary.flush()
        tokens = np.memmap(binary, mode="r", dtype=np.uint16, shape=(offsets[-1],))
        np.save(root / f"{split}-tokens.npy", tokens)
        del tokens
    np.save(root / f"{split}-offsets.npy", np.asarray(offsets, dtype=np.int32))
    story_count = len(offsets) - 1
    print(f"{split}: {story_count:,} stories, {offsets[-1]:,} tokens", flush=True)
    return story_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--cache", type=Path, help="Directory shared with previously downloaded shards")
    parser.add_argument("--tokenizer", type=Path, help="Reuse an existing BPE tokenizer")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    seen = set()
    revision = config["dataset_revision"]
    cache = args.cache or args.data
    args.data.mkdir(parents=True, exist_ok=True)
    validation = texts(shard(cache, "validation", revision), config["validation_stories"], seen)
    limit = config["train_stories"]
    train = islice(training_texts(cache, revision, seen), limit)
    if args.tokenizer:
        tokenizer = Tokenizer.from_file(str(args.tokenizer))
    else:
        train = list(train)
        tokenizer = Tokenizer(models.BPE())
        tokenizer.train_from_iterator(train, trainers.BpeTrainer(
            vocab_size=VOCAB, initial_alphabet=list(ASCII), special_tokens=SPECIAL,
            max_token_length=MAX_PIECE,
        ), length=len(train))
    if tokenizer.get_vocab_size() != VOCAB:
        raise ValueError("BPE vocabulary did not reach its target size")
    tokenizer.encode_special_tokens = True
    tokenizer_path = args.data / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    train_count = encode_split(args.data, "train", tokenizer, train)
    if limit is not None and train_count != limit:
        raise ValueError(f"Only {train_count:,} eligible training stories, expected {limit:,}")
    encode_split(args.data, "validation", tokenizer, validation)
    (args.data / "prepared.json").write_text(json.dumps({
        "revision": revision, "train_stories": train_count, "validation_stories": len(validation),
        "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
