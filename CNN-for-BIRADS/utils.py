import numpy as np
from PIL import Image
import imageio
import matplotlib.pyplot as plt


def load_images(image_path, view, input_size):
    """
    Function that loads and preprocess input images
    :param image_path: base path to image
    :param view: L-CC / R-CC / L-MLO / R-MLO
    :param input_size: desired input size for the images (height, width)
    :return: Batch x Height x Width x Channels array
    """
    image = imageio.imread(image_path + view + '.png')
    image = Image.fromarray(image).convert("L")  # Convert to grayscale
    image = image.resize(input_size, Image.LANCZOS)  # Resize image
    image = np.array(image).astype(np.float32)
    normalize_single_image(image)
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=3)

    return image


def normalize_single_image(image):
    """
    Normalize image in-place
    :param image: numpy array
    """
    image -= np.mean(image)
    image /= np.std(image)
    plt.imshow(image, cmap='gray')  
    plt.show()
