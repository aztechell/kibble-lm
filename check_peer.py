"""Verify the export against HackyLens's production C decoder on the host."""

import argparse
import ctypes as C
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hackylens", type=Path, default=Path("../hackylens"))
    parser.add_argument("--work", type=Path, default=Path("runs/export"))
    parser.add_argument("--package", type=Path, default=Path("dist/hackylens.kmodels/kibblelm/cpu.bin"))
    parser.add_argument("--cc", type=Path)
    args = parser.parse_args()
    compiler = args.cc or args.hackylens / "_deps/host-gcc-13.2.0/mingw64/bin/gcc.exe"
    dll = args.work.resolve() / "kibble_decode.dll"
    source = args.hackylens / "firmware/src/core/kibble_decode.c"
    subprocess.run([str(compiler), "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-shared",
                    str(source), "-o", str(dll)], check=True)
    core = C.CDLL(str(dll))
    core.kibble_tokenize.argtypes = (C.c_void_p, C.c_char_p, C.POINTER(C.c_uint16))
    core.kibble_tokenize.restype = C.c_int
    core.kibble_input.argtypes = (C.c_void_p, C.POINTER(C.c_uint16), C.POINTER(C.c_uint8))
    core.kibble_input.restype = None
    core.kibble_sample.argtypes = (C.c_void_p, C.POINTER(C.c_float), C.POINTER(C.c_uint32))
    core.kibble_sample.restype = C.c_uint16
    core.kibble_append.argtypes = (C.POINTER(C.c_uint16), C.c_uint16)
    core.kibble_append.restype = None
    cpu = C.create_string_buffer(args.package.read_bytes())
    fixtures = json.loads((args.work / "fixtures.json").read_text())
    for fixture in fixtures:
        prompt = fixture["prompt"].encode("ascii")
        expected_context = fixture["tokens"]
        expected_input = (args.work / fixture["input"]).read_bytes()
        context = (C.c_uint16 * 64)()
        assert core.kibble_tokenize(cpu, prompt, context), fixture
        for token in fixture.get("history", []):
            core.kibble_append(context, token)
        assert context[:len(expected_context)] == expected_context, fixture
        tensor = (C.c_uint8 * (32 * 64))()
        core.kibble_input(cpu, context, tensor)
        assert bytes(tensor)[:len(expected_input)] == expected_input, fixture
        raw_output = (args.work / fixture["output"]).read_bytes()
        output = (C.c_float * (len(raw_output) // C.sizeof(C.c_float))).from_buffer_copy(raw_output)
        seed = C.c_uint32(210)
        token = core.kibble_sample(cpu, output, C.byref(seed))
        assert (token, seed.value) == (fixture["next"], fixture["next_seed"]), fixture
        print(f"{fixture['prompt']!r}: token {token}, input and logits pipeline match")
    for invalid in (b"A" * 25, b"A\x7f", b"One\nmore"):
        assert not core.kibble_tokenize(cpu, invalid, (C.c_uint16 * 64)())


if __name__ == "__main__":
    main()
