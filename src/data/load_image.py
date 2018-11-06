from PIL import Image, ImageEnhance
import pygame
import numpy as np

def prepare_image(filename):

    image = Image.open(filename)

    image = ImageEnhance.Contrast(image).enhance(2.0)

    image = image.convert('LA').convert('RGB')

    size = image.size

    n = np.min(size)
    c = np.array(size)/2.
    cr1 = np.floor(c-n/2.).astype(int)
    cr2 = np.ceil(c+n/2.).astype(int)

    image = image.crop(cr1.tolist()+cr2.tolist())
    return to_pygame_image(image)

def to_pygame_image(image):
    size = image.size

    mode = image.mode
    data = image.tobytes()

    py_image = pygame.image.fromstring(data, size, mode)
    return py_image