import torch
from torch import nn

VOCAB = 2048
PAD, BOS, EOS = range(3)
HEIGHT, WIDTH = 4, 8
CONTEXT = HEIGHT * WIDTH
EMBED, FEATURES = 32, 64
CENTER = (2, 3)
SNAKE = tuple(
    (row * WIDTH + (WIDTH - 1 - col if row % 2 else col) + 12) % CONTEXT
    for row in range(HEIGHT) for col in range(WIDTH)
)
PROMPTS = ("Once upon a time", "A little dog", "One sunny day", "In a small town")


class KibbleLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB, EMBED)
        self.register_buffer("snake", torch.tensor(SNAKE), persistent=False)
        layers = []
        channels = (EMBED, 48, 64, 64, FEATURES)
        for in_channels, out_channels in zip(channels, channels[1:]):
            layers.extend((nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                           nn.BatchNorm2d(out_channels), nn.ReLU()))
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(FEATURES, VOCAB)

    def embeddings(self):
        return self.embedding.weight.sigmoid()

    def inputs(self, context):
        embedding = self.embedding(context.index_select(-1, self.snake)).sigmoid()
        return embedding.reshape(-1, HEIGHT, WIDTH, EMBED).permute(0, 3, 1, 2)

    def forward(self, context):
        y, x = CENTER
        return self.head(self.backbone(self.inputs(context))[:, :, y, x])

    def fused_convolutions(self):
        return [nn.utils.fuse_conv_bn_eval(conv, bn)
                for conv, bn in zip(self.backbone[::3], self.backbone[1::3])]
