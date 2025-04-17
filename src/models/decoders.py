import torch.nn as nn


class InnerProductDecoder(nn.Module):
    def __init__(self, node_dropout):
        super().__init__()

        self.sig = nn.Sigmoid()
        self.drop = nn.Dropout(node_dropout)

    def forward(self, src, dst, h):
        return self.sig((self.drop(h[src]) * self.drop(h[dst])).sum(dim=1))
