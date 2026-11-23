# KibbleLM

KibbleLM is a **314K-parameter convolutional language model** for English story continuation on the Kendryte K210. It predicts the next subword from a 32-token window. Its four-layer backbone compiles to KModel v4 and runs on the K210 KPU.

[Download the model package](https://github.com/aztechell/kibble-lm/releases/download/v0.1.0/kibblelm-sd-v0.1.0.zip) · [Model card](docs/model-card.md) · [Train and export](docs/training.md)

## Architecture

```mermaid
flowchart LR
    tokens["32 BPE tokens"] --> grid["4 × 8 embedding grid"]
    grid --> backbone["Four 3×3 KPU convolutions"]
    backbone --> head["INT8 CPU projection"]
    head --> sample["Next token"]
```

The [PyTorch model](model.py) maps the tokens to 32-channel embeddings and reads the output at one grid position. A CPU projection scores the 2,048-token vocabulary from that position's 64 features. The export contains a **116.5 KiB** KModel backbone and a companion file for the tokenizer, embeddings, and INT8 projection. Together they take **374 KiB**.

## Training and quality

The model was trained from scratch on 100,000 TinyStories. Perplexity is measured on the same 4,096 held-out targets for all three variants.

| Model | Perplexity ↓ |
| --- | ---: |
| FP32 CNN | 48.71 |
| KModel backbone + INT8 CPU projection | 53.39 |
| Smoothed bigram baseline | 64.18 |

The [training guide](docs/training.md) covers data preparation, FP32 training, KPU calibration, and distillation of the CPU projection. It also shows how to verify the compiled graph with the nncase simulator on a PC.

## Hardware demo

The HackyLens app runs the exported model on HUSKYLENS SEN0305 and displays the continuation on the LCD. Its benchmark measured **13.2 tokens/s** with display updates and **60–63 ms** to the first token. [Benchmark setup](docs/model-card.md#hardware-measurements)

<img src="docs/images/kibblelm-lcd.png" width="320" height="240" alt="KibbleLM continuing a story on the HUSKYLENS LCD">

[Try the HUSKYLENS demo](docs/firmware.md#flash-the-release) · [Build the integration](docs/firmware.md#build-the-firmware)

KibbleLM source code is available under the [MIT License](LICENSE).
