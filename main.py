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
import msvcrt
import json
import pathlib
from PIL import Image

import numpy as np

log = logging.getLogger(__name__)

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)


    filename = "data/raw/Tobias_Schoch_TOS_kopf_big_2.jpg"

    input_file = pathlib.Path(filename)

    img = prepare_image(filename)

    img = img.resize((400, 400), Image.ANTIALIAS)
    img = pil_to_ndarray(img)

    print(skimage.io.available_plugins)

    size = img.shape[:2]
    size = np.array(size)
    N = 256
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

    i = 0
    while node is not None:
        sequence.append(node)
        i += 1
        node, target_value = next_step(state, node, target_value, img, nodes, p1, thread_color=thread_color)
        if msvcrt.kbhit():
            key = msvcrt.getch()
            if key == b'q':
                break
        print(json.dumps({'progress':int(target_value), 'nodes':len(sequence)}))
    log.info("found: {}".format(sequence))

    # store sequence
    model_file = pathlib.Path('models/').joinpath(input_file.name).with_suffix('.py')
    with open(model_file,'w+') as fp:
        fp.write("sequence = {}\n".format(sequence))
        fp.write("N = {}\n".format(N))

    # from models.philipp import sequence
    # draw_sequence(state, sequence, N, thread_color=thread_color)

    title = "{}\n{} nodes, {} edges".format(input_file.name, N, len(sequence))

    viewer2 = ImageViewer(state)
    viewer2.ax.set_title(title)
    viewer2.fig.savefig(pathlib.Path('data/processed/').joinpath(input_file.name).with_suffix('.png'))
    viewer2.fig.tight_layout()

    viewer2.show()


    viewer = ImageViewer(img)
    viewer.show()


    # pygame.display.init()
    # screen = pygame.display.set_mode(size)
    # screen.blit(to_pygame_image(img), [0, 0])
    # pygame.display.set_caption("Example code for the draw module")
    # show()

#    draw_sequence(screen, np.random.random_integers(0,127,100),128)
