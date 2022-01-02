import logging

from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from torch import Tensor

import einops

logger = logging.getLogger()


class ImagePreprocessor:
    @staticmethod
    def preprocess(image: Image, width: int = 224, height: int = 224) -> Tensor:
        logger.debug("Shape after processing: {shape}".format(shape=image.size))
        transform = Compose([Resize((width, height)), ToTensor()])
        processed_image = transform(image)
        processed_image = processed_image.unsqueeze(0)  # add batch dim
        logger.debug(
            "Shape after processing: {shape}".format(shape=processed_image.shape)
        )

        return processed_image

    @staticmethod
    def get_patches(image: Tensor, patch_size: int = 16) -> Tensor:

        return einops.rearrange(
            tensor=image,
            pattern="b c (h s1) (w s2) -> b (h w) (s1 s2 c)",
            s1=patch_size,
            s2=patch_size,
        )
