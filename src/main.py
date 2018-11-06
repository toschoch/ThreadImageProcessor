#!/usr/bin/python
# -*- coding: UTF-8 -*-
# created: 06.11.2018
# author:  TOS

import logging
from src.data.load_image import prepare_image, to_pygame_image
from src.features.manipulate_array import extract_canvas_circle
from src.visualization.draw import node_positions, draw_sequence
from skimage.viewer import ImageViewer
from skimage.io._plugins.pil_plugin import pil_to_ndarray

from src.models.loss import p1, p2
from src.models.greedy import next_step

import logging
import skimage.io

import numpy as np

log = logging.getLogger(__name__)

if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG)

    img = prepare_image("../data/raw/Walter_Huber_WHU_kopf.jpg")
    img = pil_to_ndarray(img)

    print(skimage.io.available_plugins)

    size = img.shape[:2]
    size = np.array(size)
    N = 128
    thread_color = 0.2

    nodes = node_positions(N, np.min(size) / 2, size / 2)
    nodes[:,0] = np.clip(nodes[:,0], 0, size[0]-1)
    nodes[:,1] = np.clip(nodes[:,1], 0, size[1]-1)

    state = np.zeros_like(img, dtype=img.dtype)
    state[:] = 255

    # screen.blit(img, [0,0])
    extract_canvas_circle(img)

    target_value = p1(state, img)
    node = np.random.randint(0,N-1)
    sequence = []

    while node is not None:
        sequence.append(node)
        node, target_value = next_step(state, node, target_value, img, nodes, p1, thread_color=thread_color)

    #sequence = np.random.random_integers(0,N-1,50)
    log.info("found: {}".format(sequence))
    #
    #img[:] = 255
    #draw_sequence(img, sequence, N, thread_color=thread_color)

    viewer = ImageViewer(img)
    viewer2 = ImageViewer(state)
    viewer2.show()
    viewer.show()


    # pygame.display.init()
    # screen = pygame.display.set_mode(size)
    # screen.blit(to_pygame_image(img), [0, 0])
    # pygame.display.set_caption("Example code for the draw module")
    # show()

#    draw_sequence(screen, np.random.random_integers(0,127,100),128)
