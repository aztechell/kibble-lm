"""Compile and validate the KibbleLM backbone, then package it for HackyLens."""

import argparse
import hashlib
import json
import math
import struct
import subprocess
import zlib
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

INPUT = (1, 32, 4, 8)
OUTPUT = (1, 64, 4, 8)
CHANNELS = (32, 48, 64, 64, 64)
CENTER = (2, 3)
CONV_OP = 0x2002
MODEL_BUDGET = 512 * 1024


def write_onnx(path, weights):
    nodes, tensors = [], []
    previous = "input"
    for i, (weight, bias) in enumerate(weights):
        w, b, conv = f"weight_{i}", f"bias_{i}", f"conv_{i}"
        tensors.extend((numpy_helper.from_array(weight, w), numpy_helper.from_array(bias, b)))
        nodes.append(helper.make_node("Conv", [previous, w, b], [conv],
                                      kernel_shape=[3, 3], pads=[1, 1, 1, 1]))
        previous = "output" if i == len(weights) - 1 else f"relu_{i}"
        nodes.append(helper.make_node("Relu", [conv], [previous]))
    graph = helper.make_graph(nodes, "KibbleLM", [helper.make_tensor_value_info("input", TensorProto.FLOAT, INPUT)],
                              [helper.make_tensor_value_info("output", TensorProto.FLOAT, OUTPUT)], tensors)
    model = helper.make_model(graph, ir_version=6, opset_imports=[helper.make_opsetid("", 11)])
    onnx.checker.check_model(model)
    onnx.save(model, path)


def inspect_model(path):
    data = path.read_bytes()
    magic, version, flags, target, constants, main, nodes, inputs, outputs, _ = struct.unpack_from("<10I", data)
    if (magic, version, target, inputs, outputs) != (0x4B4D444C, 4, 1, 1, 1):
        raise ValueError("Expected one-input, one-output K210 KModel v4")
    in_memory, in_type, in_offset, in_bytes = struct.unpack_from("<4I", data, 40)
    out_memory, out_type, out_offset, out_bytes = struct.unpack_from("<4I", data, 72)
    physical_bytes = 2048 if in_memory == 2 else math.prod(INPUT)
    if (in_type, in_bytes) != (1, physical_bytes) or struct.unpack_from("<4I", data, 56) != INPUT:
        raise ValueError(f"Unexpected UINT8 input contract: {in_type=}, {in_bytes=}")
    if in_memory not in (1, 2) or in_offset + in_bytes > (main if in_memory == 1 else 2 << 20):
        raise ValueError("Invalid input memory range")
    if (out_memory, out_type, out_bytes) != (1, 0, math.prod(OUTPUT) * 4) or out_offset + out_bytes > main:
        raise ValueError(f"Unexpected F32 output contract: {out_type=}, {out_bytes=}")
    headers = [struct.unpack_from("<2I", data, 88 + constants + 8 * i) for i in range(nodes)]
    if 88 + constants + 8 * nodes + sum(size for _, size in headers) != len(data):
        raise ValueError("Invalid KModel length")
    ops = [op for op, _ in headers]
    if ops.count(CONV_OP) != 4 or any(op in (2, 15, 0x1001, 0x1002, 0x1004, 0x1005) for op in ops):
        raise ValueError(f"Expected four KPU convolutions: {[hex(op) for op in ops]}")
    return dict(version=version, flags=flags, arch=target, nodes=nodes, main_memory=main,
                input_bytes=math.prod(INPUT), input_memory_bytes=in_bytes, output_bytes=out_bytes,
                operators=[hex(op) for op in ops],
                input_shape=INPUT, output_shape=OUTPUT, model_bytes=len(data),
                crc32=zlib.crc32(data), sha256=hashlib.sha256(data).hexdigest())


def raw_samples(path, samples, dtype):
    path.mkdir()
    for i, sample in enumerate(samples):
        sample.astype(dtype).tofile(path / f"{i:06}.bin")


def run_ncc(ncc, arguments, cwd, log):
    with (cwd / log).open("wb") as stream:
        result = subprocess.run([str(ncc), *arguments], cwd=cwd, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"ncc exited with {result.returncode}. See {cwd / log}")


def compile_backbone(ncc, work, model, calibration, calibrate_method="no_clip"):
    write_onnx(work / "backbone.onnx", model)
    raw_samples(work / "calibration", calibration, "<f4")
    run_ncc(ncc, ["compile", "backbone.onnx", "model.kmodel", "-i", "onnx", "-t", "k210",
                  "--inference-type", "uint8", "--input-type", "uint8", "--dataset-format", "raw",
                  "--dataset", "calibration", "--calibrate-method", calibrate_method,
                  "--max-allocator-solve-secs", "0", "--dump-ir"], work, "compile.log")
    contract = inspect_model(work / "model.kmodel")
    print(json.dumps(contract), flush=True)
    return contract


def infer(ncc, work, tensors, prefix):
    raw_samples(work / f"{prefix}-inputs", np.rint(tensors * 255).clip(0, 255), np.uint8)
    run_ncc(ncc, ["infer", "model.kmodel", f"{prefix}-outputs", "--dataset", f"{prefix}-inputs",
                  "--dataset-format", "raw"], work, f"{prefix}.log")
    outputs = sorted((work / f"{prefix}-outputs").glob("*.bin"))
    if len(outputs) != len(tensors):
        raise ValueError(f"Expected {len(tensors)} inference outputs, got {len(outputs)}")
    return np.stack([np.fromfile(path, dtype="<f4").reshape(OUTPUT[1:]) for path in outputs])


def smoke(ncc, work):
    from onnx.reference import ReferenceEvaluator

    rng = np.random.default_rng(210)
    weights = []
    for a, b in zip(CHANNELS, CHANNELS[1:]):
        bound = math.sqrt(6 / (9 * a))
        weights.append((rng.uniform(-bound, bound, (b, a, 3, 3)).astype(np.float32), np.zeros(b, np.float32)))
    tensors = rng.integers(0, 256, size=(16, *INPUT[1:]), dtype=np.uint8).astype(np.float32) / 255
    compile_backbone(ncc, work, weights, tensors)
    actual = infer(ncc, work, tensors[-3:], "smoke")
    reference = ReferenceEvaluator(str(work / "backbone.onnx"))
    expected = np.concatenate([reference.run(None, {"input": tensor[None]})[0] for tensor in tensors[-3:]])
    relative_mae = np.abs(actual - expected).mean() / np.abs(expected).mean()
    print(f"Smoke relative MAE: {relative_mae:.4f}", flush=True)
    if not relative_mae < 0.15:
        raise ValueError("KModel output does not match the float reference")


def quantized_head(weight, bias):
    scale = np.maximum(np.abs(weight).max(axis=1) / 127, np.finfo(np.float32).tiny)
    quantized = np.clip(np.rint(weight / scale[:, None]), -127, 127).astype(np.int8)
    return quantized, scale, bias


def head_logits(features, head):
    weights, scale, bias = head
    maximum = features.max(axis=-1, keepdims=True)
    feature_scale = np.where(maximum > 0, maximum / np.float32(255), np.float32(1))
    quantized = np.floor((features / feature_scale).clip(0, 255) + np.float32(0.5)).astype(np.int32)
    return ((quantized @ weights.T.astype(np.int32)).astype(np.float32) * feature_scale) * scale + bias


def sample(logits, seed):
    top = np.argsort(-logits[2:], kind="stable")[:16] + 2
    weights = np.exp((logits[top] - logits[top[0]]) / np.float32(0.85)).cumsum()
    seed = seed or 210
    seed ^= (seed << 13) & 0xFFFFFFFF
    seed ^= seed >> 17
    seed ^= (seed << 5) & 0xFFFFFFFF
    threshold = np.float32(seed >> 8) * np.float32(1 / 16777216) * weights[-1]
    return int(top[min(int(np.searchsorted(weights, threshold, side="right")), 15)]), seed


def write_cpu(path, model_crc, tokenizer_path, embeddings, head):
    vocab = json.loads(tokenizer_path.read_text(encoding="utf-8"))["model"]
    token_ids = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2}
    if any(vocab["vocab"].get(piece) != i for piece, i in token_ids.items()):
        raise ValueError("Unexpected special-token IDs")
    pieces = [b""] * 2048
    for piece, i in vocab["vocab"].items():
        if i >= 3:
            pieces[i] = piece.encode("ascii")
            if not 0 < len(pieces[i]) <= 16:
                raise ValueError(f"Invalid token length for {piece!r}")
    char_ids = [vocab["vocab"][chr(ch)] for ch in range(32, 127)]
    merges = []
    for rank, pair in enumerate(vocab["merges"]):
        a, b = pair
        merges.append((vocab["vocab"][a], vocab["vocab"][b], vocab["vocab"][a + b], rank))
    if len(merges) != 1950:
        raise ValueError(f"Unexpected merge count: {len(merges)}")
    weights, scale, bias = head
    data = (struct.pack("<4sI", b"KLB1", model_crc) + b"".join(x.ljust(17, b"\0") for x in pieces)
            + struct.pack("<96H", *char_ids, 0) + b"".join(struct.pack("<4H", *m) for m in sorted(merges))
            + embeddings.tobytes() + weights.tobytes() + scale.astype("<f4").tobytes() + bias.astype("<f4").tobytes())
    if len(data) != 263608:
        raise ValueError(f"Invalid CPU package size: {len(data)}")
    path.write_bytes(data)
    return {"bytes": len(data), "crc32": zlib.crc32(data), "sha256": hashlib.sha256(data).hexdigest()}


def write_contract(path, model, cpu):
    path.write_text(f"""#ifndef HK_KIBBLE_MODEL_CONTRACT_H
#define HK_KIBBLE_MODEL_CONTRACT_H

/* Generated by KibbleLM export.py. Pair with the matching files on SD. */
#define KIBBLE_MODEL_CRC 0x{model['crc32']:08x}U
#define KIBBLE_CPU_CRC 0x{cpu['crc32']:08x}U
#define KIBBLE_MODEL_BYTES {model['model_bytes']}U
#define KIBBLE_MODEL_FLAGS {model['flags']}U
#define KIBBLE_MODEL_NODES {model['nodes']}U
#define KIBBLE_MODEL_MAIN {model['main_memory']}U
#define KIBBLE_MODEL_INPUT_MEMORY {model['input_memory_bytes']}U

#endif
""", encoding="ascii", newline="\n")


def match_features(quantized, reference):
    """Fit an affine correction on training features and fold it into the head."""
    actual, expected = quantized.astype(np.float64), reference.astype(np.float64)
    origin, center = actual.mean(0), expected.mean(0)
    actual -= origin
    expected -= center
    covariance = actual.T @ actual
    ridge = 0.001 * np.trace(covariance) / len(center)
    mapping = np.linalg.solve(covariance + ridge * np.eye(len(center)), actual.T @ expected)
    return mapping.astype(np.float32), (center - origin @ mapping).astype(np.float32)


def distill_head(model, actual, reference):
    """Recover quantization accuracy while retaining the same INT8 CPU head."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(213)
    mapping, offset = match_features(actual, reference)
    prior_weights = model.head.weight
    projection = torch.nn.Linear(64, 2048).to(device)
    with torch.no_grad():
        projection.weight.copy_(prior_weights @ torch.from_numpy(mapping.T))
        projection.bias.copy_(model.head.bias + prior_weights @ torch.from_numpy(offset))
        teacher = model.head(torch.from_numpy(reference)).to(device).softmax(-1)
        inputs = torch.from_numpy(actual).to(device)
        scale = (inputs.amax(-1, keepdim=True) / 255).clamp(min=1e-12)
        inputs = (inputs / scale + 0.5).floor().clip(0, 255) * scale
        weight_average = projection.weight.detach().clone()
        bias_average = projection.bias.detach().clone()

    optimizer = torch.optim.AdamW(projection.parameters(), lr=0.003)
    with torch.enable_grad():
        for epoch in range(1, 16):
            optimizer.param_groups[0]["lr"] = 0.003 * (36 - epoch) / 35
            for batch in torch.randperm(len(inputs), device=device, generator=generator).split(256):
                optimizer.zero_grad(set_to_none=True)
                logits = projection(inputs[batch]).log_softmax(-1)
                loss = -(teacher[batch] * logits).sum(-1).mean()
                loss.backward()
                optimizer.step()
                weight_average.lerp_(projection.weight.detach(), 0.025)
                bias_average.lerp_(projection.bias.detach(), 0.025)
    return weight_average.cpu().numpy(), bias_average.cpu().numpy()


def export_trained(args):
    import shutil

    import torch
    import torch.nn.functional as F
    from tokenizers import Tokenizer

    from model import BOS, CONTEXT, EOS, PAD, KibbleLM
    from train import Corpus

    torch.set_num_threads(4)
    ncc, work = args.ncc.resolve(), args.work.resolve()
    run = args.checkpoint.parent
    model = KibbleLM().eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    with torch.no_grad():
        training = Corpus(args.data, "train", "cpu")
        validation = Corpus(args.data, "validation", "cpu")
        rng = torch.Generator().manual_seed(211)
        calibration = model.inputs(training.batch(training.sample(512, rng))[0]).numpy()
        indices = torch.from_numpy(np.load(run / "validation-indices.npy")[:args.validation_tokens])
        context, targets = validation.batch(indices)
        tensors = model.inputs(context).numpy()
        embeddings = np.rint(model.embeddings().numpy() * 255).astype(np.uint8)
        weights = [(layer.weight.numpy(), layer.bias.numpy()) for layer in model.fused_convolutions()]
        model_info = compile_backbone(ncc, work, weights, calibration, args.calibrate_method)
        alignment_context = training.batch(training.sample(8192, rng))[0]
        training_tensors = model.inputs(alignment_context)
        target_features = np.concatenate([
            model.backbone(batch)[:, :, CENTER[0], CENTER[1]].numpy()
            for batch in training_tensors.split(512)
        ])
        actual_features = infer(ncc, work, training_tensors.numpy(), "head")[:, :, CENTER[0], CENTER[1]]
        head = quantized_head(*distill_head(model, actual_features, target_features))
        actual = infer(ncc, work, tensors, "validation")[:, :, CENTER[0], CENTER[1]]
        logits = head_logits(actual, head)
        fp32_loss = float(F.cross_entropy(model(context), targets))
        quantized_loss = float(F.cross_entropy(torch.from_numpy(logits), targets))
        baseline_loss = float(-training.bigrams()[context[:, -1].long(), targets].mean())
        report = {
            "calibration": args.calibrate_method,
            "head_distillation_tokens": len(alignment_context),
            "validation_tokens": len(indices), "float_perplexity": math.exp(fp32_loss),
            "quantized_perplexity": math.exp(quantized_loss), "bigram_perplexity": math.exp(baseline_loss),
            "quantization_perplexity_ratio": math.exp(quantized_loss - fp32_loss),
        }
        tokenizer_path = args.data / "tokenizer.json"
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        tokenizer.encode_special_tokens = True
        fixtures = []
        prompts = ("", "Once upon a time", "A little dog", "Hello, world!", "aaaaaaaaaaaaaaaaaaaaaaaa", "<EOS>")
        for i, prompt in enumerate(prompts):
            tokens = [BOS, *tokenizer.encode(prompt).ids]
            ctx = ([PAD] * CONTEXT + tokens)[-CONTEXT:]
            input_float = model.inputs(torch.tensor([ctx]))[0].numpy()
            tensor_file = f"fixture-{i}.u8"
            np.rint(input_float * 255).astype(np.uint8).tofile(work / tensor_file)
            peer = infer(ncc, work, input_float[None], f"fixture-{i}")
            peer_file = f"fixture-{i}.f32"
            peer.astype("<f4").tofile(work / peer_file)
            next_id, seed = sample(head_logits(peer[:, :, CENTER[0], CENTER[1]], head)[0], 210)
            fixtures.append(dict(prompt=prompt, tokens=ctx, input=tensor_file, output=peer_file,
                                 next=next_id, next_seed=seed, eof=next_id == EOS))
        (work / "fixtures.json").write_text(json.dumps(fixtures, indent=2) + "\n")

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    destination = out / "hackylens.kmodels" / "kibblelm"
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(work / "model.kmodel", destination / "model.kmodel")
    cpu_info = write_cpu(destination / "cpu.bin", model_info["crc32"], tokenizer_path, embeddings, head)
    report["package_bytes"] = model_info["model_bytes"] + cpu_info["bytes"]
    if report["package_bytes"] > MODEL_BUDGET:
        raise ValueError(f"Package exceeds {MODEL_BUDGET} bytes")
    write_contract(out / "kibble_model_contract.h", model_info, cpu_info)
    manifest = {"model": model_info, "cpu": cpu_info, "evaluation": report}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ncc", type=Path, default=Path(".tools/ncc/ncc.exe"))
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/v2/best.pt"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("dist"))
    parser.add_argument("--validation-tokens", type=int, default=4096)
    parser.add_argument("--calibrate-method", choices=("no_clip", "l2"), default="no_clip")
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=False)
    if args.smoke:
        smoke(args.ncc.resolve(), args.work.resolve())
    else:
        export_trained(args)


if __name__ == "__main__":
    main()
