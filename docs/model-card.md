# KibbleLM model card

KibbleLM is a 314,336-parameter convolutional language model for short English story continuations. It predicts a next subword from a context of 32 tokens and uses a 2,048-token BPE vocabulary. The backbone targets the Kendryte K210 KPU.

## Architecture

The CPU gathers 32-channel embeddings into a 4×8 grid. The KModel v4 takes a `uint8` tensor of shape `[1, 32, 4, 8]` in NCHW layout and runs four 3×3 Conv2D + ReLU layers. Its FP32 output has shape `[1, 64, 4, 8]`. The CPU projects the 64 features at grid position `(2, 3)` into vocabulary logits using INT8 weights, then samples from the top 16 tokens. `model.kmodel` is 119,336 bytes. With `cpu.bin`, the exported model takes 382,944 bytes.

## Data and evaluation

The model was trained from scratch on 100,000 unique stories from Ronen Eldan and Yuanzhi Li's TinyStories. The validation split uses 2,000 disjoint stories. Data is obtained at revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64` under CDLA-Sharing-1.0. [Dataset at the pinned revision](https://huggingface.co/datasets/roneneldan/TinyStories/tree/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64) · [Data license](https://cdla.dev/sharing-1-0/)

| Model | Perplexity on the same 4,096 validation targets |
| --- | ---: |
| FP32 CNN | 48.71 |
| KModel v4 and INT8 CPU projection | 53.39 |
| Smoothed bigram | 64.18 |

The KModel result comes from the nncase simulator paired with the quantized CPU projection. Its perplexity is 9.6% above the FP32 checkpoint on this slice. The bigram uses the same training corpus and tokenizer. The model is tuned for simple stories in printable ASCII. Longer continuations can drift, repeat phrases, or change character names.

## Hardware measurements

Measurements and the [LCD capture](images/kibblelm-lcd.png) come from the SEN0305 benchmark build of HackyLens 0.9.0, on base revision `731ad75b9f96ca6b1f3c7c1211cb6677f09466ac`. The benchmark firmware SHA-256 was `3ac9498a95ab383b0617f1e2f6bf173bc90f08782c6f6a4904529edaddcc540f`. It used the same model package as v0.1.0. The v0.1.0 firmware build adds scroll and clear controls, held-button repeat, and word-boundary handling at the token limit.

| Prompt | Time to first token | Tokens/s with LCD updates |
| --- | ---: | ---: |
| Once upon a time | 62.7 ms | 13.19 |
| The puppy found | 60.0 ms | 13.19 |

Both runs generated 48 tokens, the app's limit for a single continuation. The app stops early on EOS and ends displayed text at a word boundary when it reaches the limit. The benchmark divides generated tokens by the app's generation elapsed time. The [board check](../board_check.py) records the same counters via debug UART.
