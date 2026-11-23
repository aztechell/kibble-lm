# KibbleLM and two sub-350K story models

[stories260K](https://huggingface.co/karpathy/tinyllamas/tree/0bd21da7698eaf29a0d7de3992de8a46ef624add/stories260K) is the Llama-style checkpoint used by the [ESP32-S3 port](https://github.com/DaveBben/esp32-llm). [MicroMixer-4-300K-TinyStories](https://huggingface.co/llaa33219/MicroMixer-4-300K-TinyStories) is about 7% smaller than KibbleLM by parameter count and uses gated dilated convolutions for causal token mixing. Both were trained on TinyStories.

| Model | Parameters | Vocabulary | Input limit | Bits/byte ↓ | FP32 checkpoint + tokenizer |
| --- | ---: | ---: | ---: | ---: | ---: |
| KibbleLM FP32 | 314,336 | 2,048 BPE | 32 tokens | 1.2768 | 1,365.0 KiB |
| stories260K | 260,032 | 512 SentencePiece | 512 tokens | 1.0113 | 1,043.7 KiB |
| MicroMixer-4-300K | 292,525 | 256 byte values | 1,024 bytes | 1.0161 | 1,171.5 KiB |

Both peers assigned higher likelihood to the reference stories. Their scores were close on this slice. KibbleLM uses a fixed 32-token window and a [374 KiB deployment package](model-card.md#architecture), with its four convolutions on the K210 KPU. All table scores are from FP32 inference on a PC. The [on-device measurements](model-card.md#hardware-measurements) cover the compiled KibbleLM package. MicroMixer's vocabulary is implemented directly in code, so it has no separate tokenizer file.

[MicroMixer-4-500K](https://huggingface.co/llaa33219/MicroMixer-4-500K-TinyStories) is another open checkpoint at 491,742 parameters. [stories511K](https://huggingface.co/millrace-ai/stories511K) is an example of a smaller download at 230.4 KiB, with 511,008 parameters packed in INT4 and compressed with XZ. Its supplied runner expands the weights to FP32 in memory. The two 500K-class models are listed for reference.

## Evaluation

The comparison uses 256 stories and 183,924 ASCII bytes from the [pinned TinyStories validation split](https://huggingface.co/datasets/roneneldan/TinyStories/tree/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64). It normalizes and deduplicates the first 2,000 eligible validation rows using KibbleLM's preparation code, then keeps complete stories that fit the peers' input limits and round-trip through all three tokenizers. This leaves 1,811 candidates, sampled with seed 210. The SHA-256 of the selected stories joined with newlines is `27a41954614ac4931456c316c382f9570546d63a7df0f895a75c20395d61c69c`.

Each model sees its native tokenization and available preceding context. The score is the sum of negative log₂ probabilities for the reference text divided by its byte count. BOS supplies the initial context. EOS and padding are excluded from the score. This puts models with different vocabularies on the same text scale. The results describe these checkpoints and their native context lengths. Training budgets differ, and overlap between the selected validation stories and the peers' training data was not audited.

## Sample continuations

All three models continued `Once upon a time` with seed 210, temperature 0.85, and top-k 16. The first complete sentence from each run is shown here. The script also records two other prompts and up to 160 bytes of continuation for each.

| Model | First sentence |
| --- | --- |
| KibbleLM FP32 | Once upon a time, a little girl who wanted to help him when he saw something strange brightly in the kitchen. |
| stories260K | Once upon a time, there was a little girl named Lily. |
| MicroMixer-4-300K | Once upon a time, there was a dog named Max. |

## Reproduce the run

Start with the [training environment and prepared KibbleLM data](training.md), including `runs/v2/best.pt`. The reported run used PyTorch 2.8.0 with CUDA 12.8. The KibbleLM FP32 checkpoint SHA-256 was `e07b5a81c437e25df9eef4dafd1f8d6cd2eb7ee6ab481e36f54807425f1067da`.

```powershell
uv pip install sentencepiece==0.2.1 safetensors==0.6.2
$peers = 'runs/peer-comparison'
New-Item -ItemType Directory -Force $peers | Out-Null
$llama = 'https://huggingface.co/karpathy/tinyllamas/resolve/0bd21da7698eaf29a0d7de3992de8a46ef624add/stories260K'
Invoke-WebRequest "$llama/stories260K.pt" -OutFile "$peers/stories260K.pt"
Invoke-WebRequest "$llama/tok512.model" -OutFile "$peers/tok512.model"
$llamaCode = 'https://raw.githubusercontent.com/karpathy/llama2.c/350e04fe35433e6d2941dce5a1f53308f87058eb/model.py'
Invoke-WebRequest $llamaCode -OutFile "$peers/llama2_model.py"
$mixer = 'https://huggingface.co/llaa33219/MicroMixer-4-300K-TinyStories/resolve/0bf9332f7a38e85a257eef597e8089b20c2197d1'
Invoke-WebRequest "$mixer/epoch_2.safetensors" -OutFile "$peers/epoch_2.safetensors"
git clone --filter=blob:none --sparse https://github.com/llaa33219/MicroMixer-4.git "$peers/MicroMixer-4"
git -C "$peers/MicroMixer-4" sparse-checkout set src
git -C "$peers/MicroMixer-4" checkout --detach 8300d64843dd9e0baa9270600632c25e5aa15dc5
.venv/Scripts/python.exe compare_peers.py
```

The [comparison script](../compare_peers.py) writes `runs/peer-comparison/comparison.json` with exact scores, checkpoint hashes, and generated samples.
