# KibbleLM model card

KibbleLM v0.2.0 is a 446,232-parameter convolutional language model for short English story continuations. It predicts a next subword from a context of 32 tokens and uses a 2,048-token BPE vocabulary. The backbone targets the Kendryte K210 KPU.

## Architecture

The CPU gathers 32-channel embeddings into a 4×8 grid. The KModel v4 takes a `uint8` tensor of shape `[1, 32, 4, 8]` in NCHW layout and runs four 3×3 Conv2D + ReLU layers with channels `32 → 64 → 96 → 96 → 76`. Its FP32 output has shape `[1, 76, 4, 8]`. The CPU projects the 76 features at grid position `(2, 3)` into vocabulary logits using INT8 weights, then samples from the top 16 tokens. `model.kmodel` takes 227,240 bytes and `cpu.bin` takes 288,184 bytes, for a total of 515,424 bytes.

## Data and evaluation

Training used all four TinyStories training shards and kept 1,798,537 unique normalized stories. It reused the v0.1 tokenizer and the same 2,000 disjoint validation stories. A four-layer Transformer teacher supplied soft targets to the widened CNN. The final stage widened the readout to 76 features and fine-tuned with simulated KPU quantization. [Training recipe](training.md) · [Dataset revision](https://huggingface.co/datasets/roneneldan/TinyStories/tree/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64) · [CDLA-Sharing-1.0](https://cdla.dev/sharing-1-0/)

| Model | Context | FP32 perplexity ↓ | Exported perplexity ↓ |
| --- | ---: | ---: | ---: |
| v0.1.0 release | 32 | 48.71 | 53.39 |
| v0.1 checkpoint with corrected export | 32 | 48.71 | 49.15 |
| Same-size CNN trained on the expanded corpus | 32 | 38.52 | 41.35 |
| Wider distilled CNN, 64 features | 32 | 33.87 | 34.19 |
| Wider distilled CNN, 64 features | 64 | 39.54 | 40.78 |
| **v0.2.0 release, 76 features with QAT** | **32** | **32.29** | **32.47** |

These results use the same 4,096 validation targets. The smoothed bigram trained on the v0.2 corpus scored 62.16. The KModel result combines the nncase simulator with the INT8 CPU projection. Its perplexity is 0.54% above the FP32 checkpoint and 39% below the v0.1 KModel. On the shared text benchmark, the FP32 model scored 1.135 bits per byte. [Peer comparison](peer-comparison.md)

The model is tuned for simple stories in printable ASCII. Longer continuations can drift, repeat phrases, or change character names.

## Hardware measurements

Both release images use HackyLens 0.9.0 at base revision `731ad75b9f96ca6b1f3c7c1211cb6677f09466ac` with the KibbleLM integration patch. The checks booted each image from RAM and loaded the v0.2 model from FAT32 microSD.

| Board | Prompt | First token | Tokens/s with LCD updates |
| --- | --- | ---: | ---: |
| HUSKYLENS SEN0305 | Once upon a time | 65.2 ms | 13.02 |
| HUSKYLENS SEN0305 | The puppy found | 62.6 ms | 13.40 |
| Sipeed Maix Cube | Once upon a time | 54.0 ms | 15.27 |
| Sipeed Maix Cube | The puppy found | 53.8 ms | 15.80 |

Each run generated 48 tokens, the app's limit for a single continuation. The app also stops on EOS and ends displayed text at a word boundary at the limit. The benchmark divides generated tokens by elapsed generation time. The board check records these counters and captures the LCD over debug UART. [Board check](../board_check.py) · [SEN0305 capture](images/kibblelm-lcd.png) · [Maix Cube capture](images/kibblelm-cube-lcd.png)

Switching to Face Detect during generation transferred the KPU lease on both boards. The KModel v4 runtime completes each KPU call synchronously, so the switch takes effect between calls. Reopening KibbleLM reproduced the custom continuation with seed 210. Both boards also produced identical continuations for each prompt. The test used UART controls.

The SEN0305 diagnostic build measured a 532,480-byte increase in SDK heap use after a cold model load. Its SHA-256 was `b68c03411f02c57bbfa1d45663b03c588d5b6de11367738a1b2e529e2c3e6f4b`.

| Release image | SHA-256 |
| --- | --- |
| HUSKYLENS SEN0305 | `0271778337f490507b302806c5b26894fd4c17dff207a855953dcff60e098842` |
| Sipeed Maix Cube | `442bba75e1b3592ff141091bad7c1575e7d6529a2d9b48df92034ac44041ad13` |
