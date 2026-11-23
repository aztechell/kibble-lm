# Run KibbleLM on K210 boards

## Flash the release

Download the firmware archive for your board and the shared microSD archive from [KibbleLM v0.2.0](https://github.com/aztechell/kibble-lm/releases/tag/v0.2.0). Keep the firmware `.bin` and its same-stem `.json` sidecar together. `hkflash.py` uses that sidecar to validate the image and match it to the board's flash layout.

Run these commands from the extracted firmware directory. Select `huskylens-sen0305` for HUSKYLENS or `sipeed-maix-cube` for Maix Cube, and set the board's COM port. The commands use the flasher and ISP stub from the pinned HackyLens tree. [HackyLens revision](https://github.com/aztechell/hackylens/tree/731ad75b9f96ca6b1f3c7c1211cb6677f09466ac)

```powershell
$board = 'huskylens-sen0305'
$port = 'COM10'
$firmware = (Resolve-Path ".\hackylens-full-$board.bin").Path
git clone https://github.com/aztechell/hackylens.git
Set-Location hackylens
git checkout --detach 731ad75b9f96ca6b1f3c7c1211cb6677f09466ac
python -m pip install pyserial
python tools/hkflash.py list
python tools/hkflash.py flash-monitor $firmware --board $board --port $port --duration 8
```

Extract `kibblelm-sd-v0.2.0.zip` to the root of a FAT32 microSD card. Open **KibbleLM** after boot. The app loads `hackylens.kmodels/kibble02/model.kmodel` and `cpu.bin` from the card.

| Control | Action |
| --- | --- |
| LEFT / RIGHT | Choose a prompt. After generation starts, tap to scroll one line or hold to keep scrolling. |
| OK | Start a continuation. |
| Hold OK | Clear the story and return to prompt selection. |
| BACK | Return to the menu. |

On Maix Cube, use the three-way control for LEFT, RIGHT, and OK, and POW for BACK.

## Build the firmware

Keep `kibble-lm` and `hackylens` as sibling directories. In a clean HackyLens checkout, apply the patch and follow the upstream [toolchain setup](https://github.com/aztechell/hackylens/tree/731ad75b9f96ca6b1f3c7c1211cb6677f09466ac#build). The patch includes the descriptor for the model in the release.

```powershell
Set-Location hackylens
git checkout --detach 731ad75b9f96ca6b1f3c7c1211cb6677f09466ac
git apply --check ../kibble-lm/integration/hackylens-k210.patch
git apply ../kibble-lm/integration/hackylens-k210.patch
python tools/bootstrap_deps.py
python tools/build_firmware.py full --board huskylens-sen0305 --require-app kibble-lm
```

The build writes the flash image and sidecar to `dist`. For a newly exported checkpoint, copy its generated `kibble_model_contract.h` into `firmware/src/core/` before building. Add the matching `model.kmodel` and `cpu.bin` to the directory specified by the manifest.

## Build for Maix Cube

Use the same pinned checkout, patch, and toolchain setup above, then select the Cube target.

```powershell
python tools/build_firmware.py full --board sipeed-maix-cube --require-app kibble-lm
```

Use the Cube-specific `dist/hackylens-full-sipeed-maix-cube.bin` and matching `.json` sidecar. [HackyLens Maix Cube instructions](https://github.com/aztechell/hackylens/blob/731ad75b9f96ca6b1f3c7c1211cb6677f09466ac/docs/MAIX_CUBE.md#build-and-artifacts)

## Debug UART

After opening the app, send `HKKL` for the selected prompt or `HKKL The puppy found` for your own prompt. UART prompts accept up to 24 printable ASCII characters. Both commands use seed 210. `HKKLINFO` reports text and timing.

The [board check](../board_check.py) captures LCD output and tests that FACE DETECT can take the KPU lease after switching out of KibbleLM. Pass the board ID and a matching raw firmware image through `--image` for a temporary RAM boot.

```powershell
Set-Location ../kibble-lm
.venv/Scripts/python.exe board_check.py --board huskylens-sen0305 --port COM10 --manifest dist-v0.2/manifest.json --out runs/board
```
