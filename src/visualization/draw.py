
import pygame
import logging
import numpy as np
log = logging.getLogger(__name__)

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

def node_positions(n, radius, center):

    deg = np.arange(n)*np.pi*2./n
    ni_x = np.round(np.cos(deg)*radius).astype(int) + center[0]
    ni_y = np.round(np.sin(deg)*radius).astype(int) + center[1]
    ni = np.vstack((ni_x, ni_y)).astype(int).T

    return ni

def draw_sequence(screen, sequence, n, thread_color = BLACK):
    """ plots a sequence into a image array"""

    size = np.array(screen.get_size())


    center = size/2.0
    radius = np.min(size)/2

    # render canvas
    pygame.draw.circle(screen, BLUE, np.floor(center).astype(int), np.floor(radius).astype(int), 1)
    ni = node_positions(n, radius, center)

    # render nails
    for i in range(n):
        pygame.draw.circle(screen, GREEN, ni[i,:], 3, 0)


    # choose some edges
    #sequence = np.random.random_integers(0,n,100)

    pygame.draw.aalines(screen, thread_color, False, ni[sequence,:], 1)

def show():
    pygame.display.flip()

    while True:
        event = pygame.event.wait()
        if event.type == pygame.KEYDOWN or event.type == pygame.QUIT:
            pygame.quit()
            break


if __name__ == '__main__':
    from src.data.load_image import prepare_image
    from src.features.manipulate_array import extract_canvas_circle
    from src.models.loss import p1, p2
    from src.models.greedy import next_step
    import logging

    logging.basicConfig(level=logging.INFO)


    img = prepare_image("../../data/raw/Walter_Huber_WHU_kopf.jpg")

    size = img.get_size()
    size = np.array(size)
    N=128
    thread_color = BLACK#(np.array(WHITE)*0.8).astype(int)

    nodes = node_positions(N, np.min(size)/2, size/2)

    screen = pygame.display.set_mode(size)

    state = screen.copy()
    state.fill(WHITE)

    screen.blit(img, [0,0])
    extract_canvas_circle(screen)

    target_value = p2(state, screen)
    node = np.random.randint(0,N-1)
    sequence = []

    while node is not None:
        sequence.append(node)
        node, target_value = next_step(state, node, target_value, screen, nodes, p2, thread_color=thread_color)

    log.info("found: {}".format(sequence))

    draw_sequence(screen, sequence, N, thread_color=thread_color)

    pygame.display.set_caption("Example code for the draw module")
    show()

#    draw_sequence(screen, np.random.random_integers(0,127,100),128)
