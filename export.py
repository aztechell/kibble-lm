"""Compile and validate the KibbleLM backbone, then package it for HackyLens."""

import argparse
import hashlib
import io
import json
import math
import re
import struct
import subprocess
import zlib
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from model import CONTEXT, EMBED, FEATURES, INNER_CHANNELS, VOCAB, WIDTH

CONV_OP = 0x2002
MODEL_BUDGET = 512 * 1024


def write_onnx(path, weights, input_shape, output_shape):
    nodes, tensors = [], []
    previous = "input"
    for i, (weight, bias) in enumerate(weights):
        w, b, conv = f"weight_{i}", f"bias_{i}", f"conv_{i}"
        tensors.extend((numpy_helper.from_array(weight, w), numpy_helper.from_array(bias, b)))
        nodes.append(helper.make_node("Conv", [previous, w, b], [conv],
                                      kernel_shape=[3, 3], pads=[1, 1, 1, 1]))
        previous = "output" if i == len(weights) - 1 else f"relu_{i}"
        nodes.append(helper.make_node("Relu", [conv], [previous]))
    graph = helper.make_graph(nodes, "KibbleLM", [helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
                              [helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)], tensors)
    model = helper.make_model(graph, ir_version=6, opset_imports=[helper.make_opsetid("", 11)])
    onnx.checker.check_model(model)
    onnx.save(model, path)


def inspect_model(path, input_shape, output_shape, convolutions):
    data = path.read_bytes()
    magic, version, flags, target, constants, main, nodes, inputs, outputs, _ = struct.unpack_from("<10I", data)
    if (magic, version, target, inputs, outputs) != (0x4B4D444C, 4, 1, 1, 1):
        raise ValueError("Expected one-input, one-output K210 KModel v4")
    in_memory, in_type, in_offset, in_bytes = struct.unpack_from("<4I", data, 40)
    out_memory, out_type, out_offset, out_bytes = struct.unpack_from("<4I", data, 72)
    # For an eight-wide grid the KPU packs four channels in each 64-byte row.
    _, channels, height, width = input_shape
    if width != WIDTH:
        raise ValueError(f"Unsupported KPU grid width {width}")
    physical_bytes = ((channels + 3) // 4) * height * 64 if in_memory == 2 else math.prod(input_shape)
    if (in_type, in_bytes) != (1, physical_bytes) or struct.unpack_from("<4I", data, 56) != input_shape:
        raise ValueError(f"Unexpected UINT8 input contract: {in_type=}, {in_bytes=}")
    if in_memory not in (1, 2) or in_offset + in_bytes > (main if in_memory == 1 else 2 << 20):
        raise ValueError("Invalid input memory range")
    if (out_memory, out_type, out_bytes) != (1, 0, math.prod(output_shape) * 4) or out_offset + out_bytes > main:
        raise ValueError(f"Unexpected F32 output contract: {out_type=}, {out_bytes=}")
    headers = [struct.unpack_from("<2I", data, 88 + constants + 8 * i) for i in range(nodes)]
    if 88 + constants + 8 * nodes + sum(size for _, size in headers) != len(data):
        raise ValueError("Invalid KModel length")
    ops = [op for op, _ in headers]
    if ops.count(CONV_OP) != convolutions or any(op in (2, 15, 0x1001, 0x1002, 0x1004, 0x1005) for op in ops):
        raise ValueError(f"Expected {convolutions} KPU convolutions: {[hex(op) for op in ops]}")
    return dict(version=version, flags=flags, arch=target, nodes=nodes, main_memory=main,
                input_bytes=math.prod(input_shape), input_memory_bytes=in_bytes, output_bytes=out_bytes,
                operators=[hex(op) for op in ops],
                input_shape=input_shape, output_shape=output_shape, model_bytes=len(data),
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


def compile_backbone(ncc, work, weights, calibration, calibrate_method="no_clip"):
    input_shape = (1, *calibration.shape[1:])
    output_shape = (1, weights[-1][0].shape[0], *calibration.shape[2:])
    write_onnx(work / "backbone.onnx", weights, input_shape, output_shape)
    raw_samples(work / "calibration", calibration, "<f4")
    run_ncc(ncc, ["compile", "backbone.onnx", "model.kmodel", "-i", "onnx", "-t", "k210",
                  "--inference-type", "uint8", "--input-type", "uint8", "--dataset-format", "raw",
                  # Firmware stores round(sigmoid(embedding) * 255). Explicit
                  # options pin the input range to match the stored bytes.
                  "--input-mean", "0", "--input-std", "1",
                  "--dataset", "calibration", "--calibrate-method", calibrate_method,
                  "--max-allocator-solve-secs", "0", "--dump-ir"], work, "compile.log")
    contract = inspect_model(work / "model.kmodel", input_shape, output_shape, len(weights))
    print(json.dumps(contract), flush=True)
    return contract


def infer(ncc, work, tensors, prefix, features):
    raw_samples(work / f"{prefix}-inputs", np.rint(tensors * 255).clip(0, 255), np.uint8)
    run_ncc(ncc, ["infer", "model.kmodel", f"{prefix}-outputs", "--dataset", f"{prefix}-inputs",
                  "--dataset-format", "raw"], work, f"{prefix}.log")
    outputs = sorted((work / f"{prefix}-outputs").glob("*.bin"))
    if len(outputs) != len(tensors):
        raise ValueError(f"Expected {len(tensors)} inference outputs, got {len(outputs)}")
    output_shape = (features, *tensors.shape[2:])
    return np.stack([np.fromfile(path, dtype="<f4").reshape(output_shape) for path in outputs])


def smoke(ncc, work, context, inner_channels, features):
    from onnx.reference import ReferenceEvaluator

    rng = np.random.default_rng(210)
    weights = []
    channels = (EMBED, *inner_channels, features)
    for a, b in zip(channels, channels[1:]):
        bound = math.sqrt(6 / (9 * a))
        weights.append((rng.uniform(-bound, bound, (b, a, 3, 3)).astype(np.float32), np.zeros(b, np.float32)))
    tensors = rng.integers(40, 211, size=(16, EMBED, context // WIDTH, WIDTH), dtype=np.uint8).astype(np.float32) / 255
    compile_backbone(ncc, work, weights, tensors)
    actual = infer(ncc, work, tensors[-3:], "smoke", features)
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


def write_cpu(path, model_crc, tokenizer_bytes, embeddings, head):
    vocab = json.loads(tokenizer_bytes)["model"]
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
    expected = 8 + VOCAB * (17 + EMBED + weights.shape[1] + 2 * 4) + 96 * 2 + 1950 * 8
    if len(data) != expected:
        raise ValueError(f"Invalid CPU package size: {len(data)}")
    path.write_bytes(data)
    return {"bytes": len(data), "crc32": zlib.crc32(data), "sha256": hashlib.sha256(data).hexdigest()}


def write_contract(path, model, cpu, layout, center, package_name="kibblelm"):
    _, _, height, width = model["input_shape"]
    mapping = ", ".join(map(str, layout))
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
#define KIBBLE_MODEL_FEATURES {model['output_shape'][1]}U
#define KIBBLE_CPU_BYTES {cpu['bytes']}U
#define KIBBLE_MODEL_GRID_WIDTH {width}U
#define KIBBLE_MODEL_GRID_HEIGHT {height}U
#define KIBBLE_MODEL_CENTER {center[0] * width + center[1]}U
#define KIBBLE_MODEL_LAYOUT {{ {mapping} }}
#define KIBBLE_MODEL_DIRECTORY "/hackylens.kmodels/{package_name}"

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


def distill_head(model, actual, reference, targets, label_weight, epochs):
    """Recover quantization accuracy while retaining the same INT8 CPU head."""
    import torch
    import torch.nn.functional as F

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(213)
    mapping, offset = match_features(actual, reference)
    prior_weights = model.head.weight
    projection = torch.nn.Linear(model.features, model.head.out_features).to(device)
    with torch.no_grad():
        projection.weight.copy_(prior_weights @ torch.from_numpy(mapping.T))
        projection.bias.copy_(model.head.bias + prior_weights @ torch.from_numpy(offset))
        teacher = model.head(torch.from_numpy(reference)).to(device).softmax(-1)
        inputs = torch.from_numpy(actual).to(device)
        targets = targets.to(device)
        scale = (inputs.amax(-1, keepdim=True) / 255).clamp(min=1e-12)
        inputs = (inputs / scale + 0.5).floor().clip(0, 255) * scale
        weight_average = projection.weight.detach().clone()
        bias_average = projection.bias.detach().clone()
        affine = (weight_average.cpu().numpy().copy(), bias_average.cpu().numpy().copy())

    optimizer = torch.optim.AdamW(projection.parameters(), lr=0.003)
    with torch.enable_grad():
        for epoch in range(epochs):
            optimizer.param_groups[0]["lr"] = 0.003 * (1 - 0.65 * epoch / epochs)
            for batch in torch.randperm(len(inputs), device=device, generator=generator).split(256):
                optimizer.zero_grad(set_to_none=True)
                weight = projection.weight
                scale = (weight.detach().abs().amax(-1, keepdim=True) / 127).clamp_min(1e-12)
                quantized = (weight / scale).round().clip(-127, 127) * scale
                effective_weight = weight + (quantized - weight).detach()
                logits = F.linear(inputs[batch], effective_weight, projection.bias).log_softmax(-1)
                soft = -(teacher[batch] * logits).sum(-1).mean()
                hard = F.nll_loss(logits, targets[batch])
                loss = (1 - label_weight) * soft + label_weight * hard
                loss.backward()
                optimizer.step()
                weight_average.lerp_(projection.weight.detach(), 0.025)
                bias_average.lerp_(projection.bias.detach(), 0.025)
    return affine, (weight_average.cpu().numpy(), bias_average.cpu().numpy())


def select_head(model, actual, reference, targets, label_weight, epochs):
    import torch
    import torch.nn.functional as F

    fit = len(actual) * 3 // 4
    affine, distilled = distill_head(model, actual[:fit], reference[:fit], targets[:fit], label_weight, epochs)
    candidates = {
        "direct": quantized_head(model.head.weight.detach().numpy(), model.head.bias.detach().numpy()),
        "affine": quantized_head(*affine),
        "distilled": quantized_head(*distilled),
    }
    losses = {
        name: float(F.cross_entropy(torch.from_numpy(head_logits(actual[fit:], head)), targets[fit:]))
        for name, head in candidates.items()
    }
    selected = min(losses, key=losses.get)
    return candidates[selected], selected, losses


def export_trained(args):
    import shutil

    import torch
    import torch.nn.functional as F
    from tokenizers import Tokenizer

    from model import BOS, EOS, PAD, KibbleLM
    from train import Corpus

    torch.set_num_threads(4)
    ncc, work = args.ncc.resolve(), args.work.resolve()
    run = args.checkpoint.parent
    run_config = run / "config.json"
    config = json.loads(run_config.read_text()) if run_config.exists() else {}
    model = KibbleLM(args.context or config.get("context", CONTEXT),
                     args.inner_channels or config.get("inner_channels", INNER_CHANNELS),
                     config.get("layout", "snake"), args.features or config.get("features", FEATURES)).eval()
    y, x = model.center
    checkpoint = args.checkpoint.read_bytes()
    tokenizer_bytes = (args.data / "tokenizer.json").read_bytes()
    model.load_state_dict(torch.load(io.BytesIO(checkpoint), map_location="cpu", weights_only=True))
    with torch.no_grad():
        training = Corpus(args.data, "train", "cpu", model.context)
        validation = Corpus(args.data, "validation", "cpu", model.context)
        rng = torch.Generator().manual_seed(211)
        calibration = model.inputs(training.batch(training.sample(args.calibration_tokens, rng))[0]).numpy()
        indices = torch.from_numpy(np.load(run / "validation-indices.npy")[:args.validation_tokens])
        context, targets = validation.batch(indices)
        tensors = model.inputs(context).numpy()
        embeddings = np.rint(model.embeddings().numpy() * 255).astype(np.uint8)
        weights = [(layer.weight.numpy(), layer.bias.numpy()) for layer in model.fused_convolutions()]
        model_info = compile_backbone(ncc, work, weights, calibration, args.calibrate_method)
        alignment_context, alignment_targets = training.batch(training.sample(args.alignment_tokens, rng))
        training_tensors = model.inputs(alignment_context)
        target_features = np.concatenate([
            model.backbone(batch)[:, :, y, x].numpy()
            for batch in training_tensors.split(512)
        ])
        actual_features = infer(ncc, work, training_tensors.numpy(), "head", model.features)[:, :, y, x]
        head, head_type, head_losses = select_head(model, actual_features, target_features, alignment_targets,
                                                  args.head_label_weight, args.head_epochs)
        actual = infer(ncc, work, tensors, "validation", model.features)[:, :, y, x]
        logits = head_logits(actual, head)
        fp32_loss = float(F.cross_entropy(model(context), targets))
        quantized_loss = float(F.cross_entropy(torch.from_numpy(logits), targets))
        baseline_loss = float(-training.bigrams()[context[:, -1].long(), targets].mean())
        report = {
            "calibration": args.calibrate_method,
            "calibration_tokens": len(calibration),
            "head_label_weight": args.head_label_weight, "head_epochs": args.head_epochs,
            "head_type": head_type, "head_selection_nll": head_losses,
            "head_distillation_tokens": len(alignment_context),
            "validation_tokens": len(indices), "float_perplexity": math.exp(fp32_loss),
            "quantized_perplexity": math.exp(quantized_loss), "bigram_perplexity": math.exp(baseline_loss),
            "quantization_perplexity_ratio": math.exp(quantized_loss - fp32_loss),
        }
        tokenizer = Tokenizer.from_str(tokenizer_bytes.decode("utf-8"))
        tokenizer.encode_special_tokens = True
        fixtures = []
        prompts = ("", "Once upon a time", "A little dog", "Hello, world!", "aaaaaaaaaaaaaaaaaaaaaaaa", "<EOS>", "")
        for i, prompt in enumerate(prompts):
            tokens = [BOS, *tokenizer.encode(prompt).ids]
            history = list(range(100, 103 + model.context)) if i == len(prompts) - 1 else []
            tokens.extend(history)
            ctx = ([PAD] * model.context + tokens)[-model.context:]
            input_float = model.inputs(torch.tensor([ctx]))[0].numpy()
            tensor_file = f"fixture-{i}.u8"
            np.rint(input_float * 255).astype(np.uint8).tofile(work / tensor_file)
            peer = infer(ncc, work, input_float[None], f"fixture-{i}", model.features)
            peer_file = f"fixture-{i}.f32"
            peer.astype("<f4").tofile(work / peer_file)
            next_id, seed = sample(head_logits(peer[:, :, y, x], head)[0], 210)
            fixtures.append(dict(prompt=prompt, history=history, tokens=ctx, input=tensor_file, output=peer_file,
                                 next=next_id, next_seed=seed, eof=next_id == EOS))
        (work / "fixtures.json").write_text(json.dumps(fixtures, indent=2) + "\n")

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    destination = out / "hackylens.kmodels" / args.package_name
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(work / "model.kmodel", destination / "model.kmodel")
    cpu_info = write_cpu(destination / "cpu.bin", model_info["crc32"], tokenizer_bytes, embeddings, head)
    report["package_bytes"] = model_info["model_bytes"] + cpu_info["bytes"]
    if report["package_bytes"] > MODEL_BUDGET:
        raise ValueError(f"Package exceeds {MODEL_BUDGET} bytes")
    write_contract(out / "kibble_model_contract.h", model_info, cpu_info, model.snake.tolist(), model.center,
                   args.package_name)
    manifest = {"package_directory": f"/hackylens.kmodels/{args.package_name}",
                "input_quantization": {"scale": 1 / 255, "zero_point": 0},
                "training_checkpoint_sha256": hashlib.sha256(checkpoint).hexdigest(),
                "tokenizer_sha256": hashlib.sha256(tokenizer_bytes).hexdigest(),
                "model": model_info, "cpu": cpu_info, "evaluation": report}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ncc", type=Path, default=Path(".tools/ncc/ncc.exe"))
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/v2/best.pt"))
    parser.add_argument("--context", type=int, choices=(32, 64))
    parser.add_argument("--inner-channels", type=int, nargs=3)
    parser.add_argument("--features", type=int, choices=(64, 72, 76))
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("dist"))
    parser.add_argument("--package-name", default="kibblelm")
    parser.add_argument("--validation-tokens", type=int, default=4096)
    parser.add_argument("--calibrate-method", choices=("no_clip", "l2"), default="no_clip")
    parser.add_argument("--calibration-tokens", type=int, default=512)
    parser.add_argument("--alignment-tokens", type=int, default=8192)
    parser.add_argument("--head-epochs", type=int, default=25)
    parser.add_argument("--head-label-weight", type=float, default=0.15)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9]{1,8}", args.package_name):
        parser.error("--package-name must contain 1 to 8 lowercase letters or digits")
    if not 0 <= args.head_label_weight <= 1:
        parser.error("--head-label-weight must be between 0 and 1")
    if args.alignment_tokens < 256 or args.calibration_tokens < 1 or args.validation_tokens < 1 or args.head_epochs < 1:
        parser.error("Calibration, evaluation and head-training counts must be positive, with at least 256 alignment tokens")
    args.work.mkdir(parents=True, exist_ok=False)
    if args.smoke:
        smoke(args.ncc.resolve(), args.work.resolve(), args.context or CONTEXT,
              args.inner_channels or INNER_CHANNELS, args.features or FEATURES)
    else:
        export_trained(args)


if __name__ == "__main__":
    main()
