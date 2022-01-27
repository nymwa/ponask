import torch.nn as nn
from .attention import SelfAttentionSubLayer
from .feedforward import FeedForwardSubLayer

class TransformerEncoderLayer(nn.Module):

    def __init__(
            self,
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            attention_dropout,
            activation_dropout):

        super().__init__()
        self.self_attn_layer = SelfAttentionSubLayer(
                d_model,
                nhead,
                dropout,
                attention_dropout)
        self.feed_forward_layer = FeedForwardSubLayer(
                d_model,
                dim_feedforward,
                activation_dropout)

    def forward(
            self,
            x,
            attn_mask = None,
            padding_mask = None):

        x = self.self_attn_layer(
                x,
                attn_mask,
                padding_mask)
        x = self.feed_forward_layer(x)
        return x

