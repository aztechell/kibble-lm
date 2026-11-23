"""Run KibbleLM hardware acceptance and capture both LCD results over UART."""

import argparse
import hashlib
import json
import re
import struct
import sys
import time
import zlib
from pathlib import Path


INFO = re.compile(r"HKKLINFO phase=(\d+) tokens=(\d+) first_us=(\d+) elapsed_us=(\d+) accelerator_us=(\d+) head_us=(\d+)")


def send(link, command):
    link.write((command + "\n").encode("ascii"))
    link.flush()


def receive(link, pattern, timeout=5):
    end = time.monotonic() + timeout
    lines = []
    while time.monotonic() < end:
        line = link.readline().decode("ascii", "replace").rstrip("\r\n")
        if line:
            lines.append(line)
            if re.fullmatch(pattern, line):
                return line
    raise TimeoutError(f"Expected {pattern!s}. Last lines: {lines[-6:]}")


def command(link, cmd, pattern, timeout=5):
    send(link, cmd)
    return receive(link, pattern, timeout)


def status(link):
    line = command(link, "HKKLINFO", INFO)
    info = dict(zip(("phase", "tokens", "first_us", "elapsed_us", "accelerator_us", "head_us"),
                    map(int, INFO.fullmatch(line).groups())))
    info["text"] = receive(link, r"HKKL TEXT .*")[len("HKKL TEXT "):]
    return info


def wait_phase(link, expected):
    end = time.monotonic() + 20
    while time.monotonic() < end:
        info = status(link)
        if info["phase"] == expected:
            return info
        if info["phase"] == 4:
            raise RuntimeError(f"KibbleLM reported error: {info}")
        time.sleep(0.10)
    raise TimeoutError(f"KibbleLM did not reach phase {expected}: {info}")


def screenshot(link, path, hkflash):
    send(link, "HKSHOT")
    header = b""
    end = time.monotonic() + 6
    while time.monotonic() < end:
        header = hkflash.read_serial_line(link, timeout=2)
        if header.startswith(b"HKSHOT BEGIN BMP24 "):
            break
    _, _, _, width, height, size = header.split()
    image = hkflash.read_exact(link, int(size), timeout=45, label="LCD screenshot")
    if image[:2] != b"BM" or struct.unpack_from("<ii", image, 18) != (int(width), int(height)):
        raise ValueError("Bad screenshot BMP header")
    footer = receive(link, r"HKSHOT END [0-9A-F]{8}")
    if int(footer.rsplit(maxsplit=1)[1], 16) != zlib.crc32(image):
        raise ValueError("LCD screenshot CRC mismatch")
    path.write_bytes(image)


def boot(link, image, hkflash, paced_isp):
    raw = image.read_bytes()
    if not 0 < len(raw) < 2 * 1024 * 1024:
        raise ValueError("Expected a raw SEN0305 RAM image below 2 MiB")
    isp = paced_isp(link)
    hkflash.connect_bootrom_uploader(isp, link, attempts=3)
    try:
        baud = 460800
        isp.write_packet(hkflash.request(0xc6, struct.pack("<III", 0, 4, round(baud * 38.6 / 38))))
        time.sleep(0.05)
        link.baudrate = baud
        isp.bootrom_greeting(retries=3, timeout=0.5)
    except (OSError, hkflash.ProtocolError):
        link.baudrate = 115200
        hkflash.connect_bootrom_uploader(isp, link, attempts=3)
    print(f"[RAM] {len(raw)} bytes @ {link.baudrate}", flush=True)
    isp.memory_write(hkflash.STUB_ADDRESS, raw)
    isp.boot_sram()
    link.baudrate = 115200
    time.sleep(1)
    for _ in range(4):
        try:
            command(link, "HKPING", r"HKPONG", timeout=2)
            return
        except TimeoutError:
            pass
    raise TimeoutError("Temporary RAM firmware did not answer HKPING")


def generate(link, cmd, path, hkflash):
    command(link, cmd, r"HKKL OK")
    info = wait_phase(link, 3)
    if not info["tokens"]:
        raise ValueError(f"The board produced no continuation: {info['text']!r}")
    info["tokens_per_second"] = info["tokens"] * 1e6 / info["elapsed_us"]
    screenshot(link, path, hkflash)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hackylens", type=Path, default=Path("../hackylens"))
    parser.add_argument("--port", default="COM10")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--out", type=Path, default=Path("runs/board"))
    parser.add_argument("--check-card", action="store_true", help="Read FAT32 mount status from the test firmware and exit")
    parser.add_argument("--fixture", action="store_true", help="Install new package files with the RAM-only provisioner")
    args = parser.parse_args()
    sys.path.insert(0, str(args.hackylens.resolve() / "tools"))
    import hkflash
    from object_gate_ram import PacedIsp

    args.out.mkdir(parents=True, exist_ok=True)
    with hkflash.open_runtime_serial(hkflash.require_serial(), args.port, 115200) as link:
        if args.image:
            boot(link, args.image, hkflash, PacedIsp)
        if args.check_card:
            print(command(link, "HKKLCARD", r"HKKLCARD present=\d mounted=\d heap=\d+"), flush=True)
            return
        report = {"board": "huskylens-sen0305"}
        if args.image:
            report["firmware_sha256"] = hashlib.sha256(args.image.read_bytes()).hexdigest()
        manifest = Path("dist/manifest.json")
        if manifest.exists():
            info = json.loads(manifest.read_text())
            report.update(model_sha256=info["model"]["sha256"], cpu_sha256=info["cpu"]["sha256"])
        if args.fixture:
            report["sd_install"] = command(link, "HKKLINSTALL", r"HKKLINSTALL .*", timeout=45)
            if not report["sd_install"].startswith("HKKLINSTALL OK"):
                raise RuntimeError(report["sd_install"])
            report["before_open"] = command(link, "HKKLCARD", r"HKKLCARD present=1 mounted=1 heap=\d+")
        command(link, "HKAPP kibble-lm", r"HKAPP OK")
        wait_phase(link, 1)
        if args.fixture:
            report["after_open"] = command(link, "HKKLCARD", r"HKKLCARD present=1 mounted=1 heap=\d+")
            before = int(report["before_open"].split("heap=")[1])
            after = int(report["after_open"].split("heap=")[1])
            report["heap_growth_bytes"] = before - after
        report["preset"] = generate(link, "HKKL", args.out / "preset.bmp", hkflash)
        report["custom"] = generate(link, "HKKL The puppy found", args.out / "custom.bmp", hkflash)
        command(link, "HKKL One sunny day", r"HKKL OK")
        if status(link)["phase"] != 2:
            raise RuntimeError("KibbleLM did not enter generation before the switch")
        # Switching uses the same runtime stop and lease retirement as BACK.
        command(link, "HKAPP face-detect", r"HKAPP OK")
        deadline = time.monotonic() + 12
        while time.monotonic() < deadline:
            face = command(link, "HKFACEINFO", r"HKFACEINFO .*", timeout=3)
            if (match := re.search(r"\bruns=(\d+)", face)) and int(match.group(1)) > 0:
                report["face_after_switch"] = face
                break
            time.sleep(0.25)
        else:
            report["face_after_switch"] = face
        command(link, "HKAPP kibble-lm", r"HKAPP OK")
        wait_phase(link, 1)
        command(link, "HKKL The puppy found", r"HKKL OK")
        repeated = wait_phase(link, 3)
        if report["custom"]["text"] != repeated["text"]:
            raise ValueError("Fixed-seed text differed after reopening the app")
        report["reopen_matches"] = True
        (args.out / "board.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
