import logging

from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor

logger = logging.getLogger()

class ImagePreprocessor:

    @staticmethod
    def preprocess(image: Image, width: int = 224, height: int = 224) -> Image:
        logger.debug("Shape after processing: {shape}".format(shape=image.shape))
        transform = Compose([Resize((width, height)), ToTensor()])
        processed_image = transform(image)
        processed_image = processed_image.unsqueeze(0)  # add batch dim
        logger.debug("Shape after processing: {shape}".format(shape=processed_image.shape))

        return processed_image

    @staticmethod
    def get_patches(image: Image, N : int) -> List[Image]:

        return []