# KibbleLM

KibbleLM predicts the next English subword from a window of 32 tokens. Four 3×3 convolutions run on the Kendryte K210 KPU. The HackyLens app tokenizes the prompt, runs an INT8 output projection on the CPU and prints the sampled continuation on the LCD.

## Training and export on Windows

Use Python 3.12, a CUDA-capable NVIDIA GPU, and the `uv` dependency manager. The dataset is [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) under CDLA-Sharing-1.0. The revision, subset sizes and seed are fixed in `config.json`. Training and validation data, checkpoints, compiler binaries and SD files are ignored by Git.

```powershell
uv venv --python 3.12
uv pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt
$python = '.venv/Scripts/python.exe'
& $python bootstrap_ncc.py
& $python export.py --smoke --work runs/smoke
& $python prepare.py
& $python train.py
& $python export.py --work runs/export-final
& $python check_peer.py --work runs/export-final
```

The best checkpoint and full float validation report are in `runs/v2`. Export uses 512 training inputs for KPU calibration and another 8,192 training inputs to distill the INT8 CPU head against the float model. The report compares the float model, compiled package and a smoothed bigram on the same 4,096 held-out tokens. The KModel inspection checks all four KPU convolution opcodes, the logical CHW input, its packed KPU memory size and the float feature output. The C parity check runs HackyLens's actual tokenizer, input builder and sampler against compiler-generated fixtures.

## HackyLens app

Keep `kibble-lm` and `hackylens` as sibling directories. Use the generated descriptor with its matching assets.

```powershell
Copy-Item dist/kibble_model_contract.h ../hackylens/firmware/src/runtime/kibble_model_contract.h
Set-Location ../hackylens
python tools/build_firmware.py full --board huskylens-sen0305 --require-app kibble-lm
```

Copy `dist/hackylens.kmodels/kibblelm` from this repository into `hackylens.kmodels` on the FAT32 card. Launch **KibbleLM** from the menu. **LEFT** and **RIGHT** select a prompt, **OK** generates a continuation and **BACK** exits. The app shows up to 48 new tokens and ends early on EOS. The display follows the latest wrapped lines.

With the app open and ready, send `HKKL The puppy found` on the debug UART to generate from up to 24 printable ASCII characters. `HKKL` uses the current built-in prompt. Both debug commands use seed 210 for repeatable checks. `HKKLINFO` reports the current text and timing.

```powershell
.venv/Scripts/python.exe board_check.py --port COM10 --out runs/board
```

The check expects matching firmware and SD files on the SEN0305. It captures both screenshots, switches to FACE DETECT during generation and checks fixed-seed output after reopening. It accepts a raw build image with `--image` for temporary RAM boot.
