from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

SCRIPT_PATH = Path(__file__).parent

if __name__ == "__main__":

    # Loading image
    image = Image.open(fp=SCRIPT_PATH.joinpath("data").joinpath("my_dog.jpg"))
    fig = plt.figure()
    plt.imshow(image)
    plt.show()

    # Preprocessing
    print("A")