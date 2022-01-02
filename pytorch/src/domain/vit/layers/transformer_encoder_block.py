from torch import nn

from src.domain.vit.layers.feed_forward_block import FeedForwardBlock
from src.domain.vit.layers.residual_add import ResidualAdd
from src.domain.vit.layers.multi_head_attention import MultiHeadAttention


class TransformerEncoderBlock(nn.Sequential):
    def __init__(
        self,
        emb_size: int = 768,
        drop_p: float = 0.0,
        forward_expansion: int = 4,
        forward_drop_p: float = 0.0,
        **kwargs
    ):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    MultiHeadAttention(emb_size, **kwargs),
                    nn.Dropout(drop_p),
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    FeedForwardBlock(
                        emb_size, expansion=forward_expansion, drop_p=forward_drop_p
                    ),
                    nn.Dropout(drop_p),
                )
            ),
        )
