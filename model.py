import torch
from torch import nn

VOCAB = 2048
PAD, BOS, EOS = range(3)
HEIGHT, WIDTH = 4, 8
CONTEXT = HEIGHT * WIDTH
EMBED, FEATURES = 32, 64
INNER_CHANNELS = (48, 64, 64)
CENTER = (2, 3)
SNAKE = tuple(
    (row * WIDTH + (WIDTH - 1 - col if row % 2 else col) + 12) % CONTEXT
    for row in range(HEIGHT) for col in range(WIDTH)
)
SNAKE_64 = (
    *range(8), *range(15, 7, -1),
    *(CONTEXT + token for token in SNAKE),
    *range(16, 24), *range(31, 23, -1),
)
PROMPTS = ("Once upon a time", "A little dog", "One sunny day", "In a small town")


def spiral_layout(height, width, center):
    """Place recent tokens within fewer convolution steps of the readout."""
    row, column = center
    cells = [row * width + column]
    directions = ((0, -1), (1, 0), (0, 1), (-1, 0))
    for turn in range(4 * max(height, width)):
        dy, dx = directions[turn % len(directions)]
        for _ in range(1 + turn // 2):
            row, column = row + dy, column + dx
            if 0 <= row < height and 0 <= column < width:
                cells.append(row * width + column)
    layout = [0] * (width * height)
    for token, cell in enumerate(reversed(cells)):
        layout[cell] = token
    return layout


class KibbleLM(nn.Module):
    def __init__(self, context=CONTEXT, inner_channels=INNER_CHANNELS, layout="snake", features=FEATURES):
        super().__init__()
        grid, self.center = {32: (SNAKE, CENTER), 64: (SNAKE_64, (4, 3))}[context]
        self.context = context
        self.height = context // WIDTH
        self.features = features
        if layout == "spiral":
            grid = spiral_layout(self.height, WIDTH, self.center)
        elif layout != "snake":
            raise ValueError(f"Unknown context layout {layout!r}")
        self.embedding = nn.Embedding(VOCAB, EMBED)
        self.register_buffer("snake", torch.tensor(grid), persistent=False)
        layers = []
        channels = (EMBED, *inner_channels, features)
        for in_channels, out_channels in zip(channels, channels[1:]):
            layers.extend((nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                           nn.BatchNorm2d(out_channels), nn.ReLU()))
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(features, VOCAB)

    def embeddings(self):
        return self.embedding.weight.sigmoid()

    def inputs(self, context):
        embedding = self.embedding(context.index_select(-1, self.snake)).sigmoid()
        return embedding.reshape(-1, self.height, WIDTH, EMBED).permute(0, 3, 1, 2)

    def forward(self, context):
        y, x = self.center
        return self.head(self.backbone(self.inputs(context))[:, :, y, x])

    def fused_convolutions(self):
        return [nn.utils.fuse_conv_bn_eval(conv, bn)
                for conv, bn in zip(self.backbone[::3], self.backbone[1::3])]
