from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
from torchsummary import summary

from src.services import ImagePreprocessor
from src.domain.vit.layers.patch_embedding import PatchEmbedding
from src.domain.vit.layers.transformer_encoder_block import TransformerEncoderBlock
from src.domain.vit.vision_transformer import ViT

SCRIPT_PATH = Path(__file__).parent

if __name__ == "__main__":
    # Implementation from https://towardsdatascience.com/implementing-visualttransformer-in-pytorch-184f9f16f632

    # Loading image
    image = Image.open(fp=SCRIPT_PATH.joinpath("data").joinpath("my_dog.jpg"))
    fig = plt.figure()
    plt.imshow(image)
    plt.show()

    # Preprocessing
    image = ImagePreprocessor.preprocess(image=image)
    # patches = ImagePreprocessor.get_patches(image=image)

    # Embedded Patches
    embedded_patches = PatchEmbedding()(image)
    encoder = TransformerEncoderBlock()(embedded_patches)


    vision_transformer = ViT()(image)
    summary(ViT(), (3, 224, 224), device='cpu')
    print("A")
