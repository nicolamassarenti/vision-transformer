from torch import nn

from src.domain.vit.layers.patch_embedding import PatchEmbedding
from src.domain.vit.layers.transformer_encoder import TransformerEncoder
from src.domain.vit.layers.classification_head import ClassificationHead

class ViT(nn.Sequential):
    def __init__(self,
                in_channels: int = 3,
                patch_size: int = 16,
                emb_size: int = 768,
                img_size: int = 224,
                depth: int = 12,
                n_classes: int = 1000,
                **kwargs):
        super().__init__(
            PatchEmbedding(in_channels=in_channels, patch_size=patch_size, emb_size=emb_size, img_size=img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )