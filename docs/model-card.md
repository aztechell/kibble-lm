# KibbleLM model card

KibbleLM continues short English stories on HUSKYLENS SEN0305. It uses a fixed context of 32 subword tokens and a vocabulary of 2,048 BPE pieces. The app generates up to 48 new tokens, stops early on EOS, and ends displayed text at a word boundary when it reaches the limit.

## Architecture

The CPU gathers embeddings and arranges the 32 context tokens as a 4×8 grid. The KModel v4 runs four 3×3 Conv2D + ReLU layers on the K210 KPU. The CPU projects 64 features at the center position into vocabulary logits using INT8 weights, then samples from the top 16 tokens. The two files on microSD occupy 382,944 bytes together.

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

Both runs generated 48 tokens. The measurement counts generated tokens across the app's generation elapsed time. The [board check](../board_check.py) records the same counters via debug UART.
