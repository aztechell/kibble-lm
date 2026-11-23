# Run KibbleLM on HUSKYLENS SEN0305

## Flash the release

Download the firmware and microSD archives from [KibbleLM v0.1.0](https://github.com/aztechell/kibble-lm/releases/tag/v0.1.0). Keep the firmware `.bin` and its same-stem `.json` sidecar together. `hkflash.py` uses that sidecar to validate the image and match it to the SEN0305 flash layout.

The following Windows commands use the flasher and ISP stub in the pinned [HackyLens repository](https://github.com/aztechell/hackylens/tree/731ad75b9f96ca6b1f3c7c1211cb6677f09466ac). Run them from the directory where you extracted the firmware and replace `COM10` with the port reported for your board.

```powershell
$firmware = (Resolve-Path .\hackylens-full-huskylens-sen0305.bin).Path
git clone https://github.com/aztechell/hackylens.git
Set-Location hackylens
git checkout --detach 731ad75b9f96ca6b1f3c7c1211cb6677f09466ac
python -m pip install pyserial
python tools/hkflash.py list
python tools/hkflash.py flash-monitor $firmware --board huskylens-sen0305 --port COM10 --duration 8
```

Extract `kibblelm-sd-v0.1.0.zip` to the root of a FAT32 microSD card. Open **KibbleLM** once the firmware has booted and the card is inserted. The app's default location is `hackylens.kmodels/kibblelm`.

## Build the firmware

Keep `kibble-lm` and `hackylens` as sibling directories. In a clean HackyLens checkout, apply the patch and follow the upstream [toolchain setup](https://github.com/aztechell/hackylens/tree/731ad75b9f96ca6b1f3c7c1211cb6677f09466ac#build). The patch includes the descriptor for the model in the release.

```powershell
Set-Location hackylens
git checkout --detach 731ad75b9f96ca6b1f3c7c1211cb6677f09466ac
git apply --check ../kibble-lm/integration/hackylens-sen0305.patch
git apply ../kibble-lm/integration/hackylens-sen0305.patch
python tools/bootstrap_deps.py
python tools/build_firmware.py full --board huskylens-sen0305 --require-app kibble-lm
```

The build writes the flash image and sidecar to `dist`. When using a newly exported checkpoint, copy its generated `kibble_model_contract.h` into `firmware/src/runtime/` before building and use the matching `model.kmodel` and `cpu.bin` on the card.

## Debug UART

After opening the app, send `HKKL` for the selected prompt or `HKKL The puppy found` for your own prompt. UART prompts accept up to 24 printable ASCII characters. Both commands use seed 210. `HKKLINFO` reports text and timing.

The [board check](../board_check.py) captures LCD output and tests that FACE DETECT can take the KPU lease after switching out of KibbleLM. Its optional `--image` argument accepts a raw SEN0305 firmware image for temporary RAM boot.

```powershell
Set-Location ../kibble-lm
.venv/Scripts/python.exe board_check.py --port COM10 --out runs/board
```
