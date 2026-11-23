"""Approximate the nncase K210 uint8 path during fine-tuning."""

import torch
import torch.nn.functional as F


CALIBRATION_TOKENS = 512
CALIBRATION_SEED = 211


def straight_through(value, quantized):
    return value + (quantized - value).detach()


def unsigned(value, limit, *, truncate=False):
    step = (limit / 255).clamp_min(1e-12)
    scaled = value / step
    codes = scaled.floor() if truncate else scaled.round()
    return straight_through(value, codes.clamp(0, 255) * step)


def kpu_weights(weight):
    # nncase v0.2 first scales every output filter to the layer's range.
    # KPU batch normalization applies the reciprocal after accumulation.
    with torch.no_grad():
        low = weight.amin((1, 2, 3), keepdim=True).clamp_max(-1e-8)
        high = weight.amax((1, 2, 3), keepdim=True).clamp_min(1e-8)
        layer_low, layer_high = low.amin(), high.amax()
        gain = torch.minimum(layer_low / low, layer_high / high)
        step = (layer_high - layer_low) / 255
        zero = (-layer_low / step).round()
        codes = (weight * gain / step + zero).round().clamp(0, 255)
        quantized = (codes - zero) * (step / gain)
    return straight_through(weight, quantized)


class KpuQat:
    def __init__(self, model, corpus):
        self.model = model
        generator = torch.Generator().manual_seed(CALIBRATION_SEED)
        count = len(corpus.tokens) - len(corpus.offsets) + 1
        indices = torch.randint(count, (CALIBRATION_TOKENS,), generator=generator)
        self.calibration, _ = corpus.batch(indices.to(corpus.tokens.device))
        self.refresh_ranges()

    @torch.no_grad()
    def refresh_ranges(self):
        self.model.eval()
        x = self.model.inputs(self.calibration)
        ranges = []
        layers = self.model.backbone
        for conv, bn, relu in zip(layers[::3], layers[1::3], layers[2::3]):
            x = relu(bn(conv(x)))
            ranges.append(x.amax().clamp_min(1e-5))
        self.ranges = ranges

    def __call__(self, context):
        model = self.model
        x = model.inputs(context)
        x = unsigned(x, x.new_tensor(1.0))
        for conv, bn, limit in zip(model.backbone[::3], model.backbone[1::3], self.ranges):
            gain = bn.weight / (bn.running_var + bn.eps).sqrt()
            weight = conv.weight * gain[:, None, None, None]
            bias = bn.bias - bn.running_mean * gain
            x = F.conv2d(x, kpu_weights(weight), bias, padding=1).float().relu()
            x = unsigned(x.clamp_max(limit), limit, truncate=True)
        row, col = model.center
        features = x[:, :, row, col]
        features = unsigned(features, features.detach().amax(-1, keepdim=True))
        weight = model.head.weight
        step = (weight.detach().abs().amax(-1, keepdim=True) / 127).clamp_min(1e-12)
        quantized = (weight / step).round().clamp(-127, 127) * step
        return F.linear(features, straight_through(weight, quantized), model.head.bias)
