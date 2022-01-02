import torch
from torch import nn, Tensor
from torch.functional import F
from einops import rearrange


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 512, num_heads: int = 8, dropout: float = 0):
        super().__init__()

        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)


    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads.
        # Resulting shape: [BATCH, HEADS, SEQUENCE_LEN, EMBEDDING_SIZE]
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)

        # To compute attention we need first to do matrix multiplication between Q and V
        # so we need to sum up over the last axis
        # Resulting shape: [BATCH, HEADS, QUERY_LEN, KEY_LEN]
        energy = torch.einsum(
            "bhqd, bhkd -> bhqk", queries, keys
        )  # batch, num_heads, query_len, key_len

        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)

        # We use attention to scale values
        # Resulting shape: [BATCH HEADS, VALUES_LEN, EMBEDDING_SIZE]
        out = torch.einsum("bhal, bhlv -> bhav ", att, values)
        # We concatenate the heads together
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)

        return out
