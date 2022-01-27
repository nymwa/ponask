import torch.nn as nn
from .encoder_layer import TransformerEncoderLayer

class TransformerEncoder(nn.Module):

    def __init__(
            self,
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            attention_dropout,
            activation_dropout,
            num_layers):

        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model,
                nhead,
                dim_feedforward,
                dropout,
                attention_dropout,
                activation_dropout)
            for _
            in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, padding_mask = None):
        for layer in self.layers:
            x = layer(x, padding_mask = padding_mask)
        x = self.norm(x)
        return x

