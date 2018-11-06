
import skimage.draw
import skimage.filters
from scipy.ndimage.filters import convolve
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

    size = np.array(screen.shape[:2])


    center = size/2.0
    radius = np.min(size)/2

    # render canvas
    center_int = np.floor(center).astype(int)
    rr, cc, val = skimage.draw.circle_perimeter_aa(center_int[0], center_int[1] ,np.floor(radius).astype(int)-1)
    screen[rr, cc, :] = val[:,None] * 255
    #pygame.draw.circle(screen, BLUE, np.floor(center).astype(int), np.floor(radius).astype(int), 1)
    ni = node_positions(n, radius, center)
    ni[:,0] = np.clip(ni[:,0], 0, size[0]-1)
    ni[:,1] = np.clip(ni[:,1], 0, size[1]-1)

    # render nails
    for i in range(n):
        rr, cc = skimage.draw.circle(ni[i,0], ni[i,1], 3)
        rr = np.clip(rr,0,screen.shape[0]-1)
        cc = np.clip(cc,0,screen.shape[1]-1)
        screen[rr, cc, :] = np.array([0, 255, 0])

    # draw edges
    for i in range(1,len(sequence)):
        rr, cc, val = skimage.draw.line_aa(ni[sequence[i-1],0], ni[sequence[i-1],1],
                                           ni[sequence[i],0], ni[sequence[i],1])
        #val = (255*val).astype(np.uint8)
        screen[rr, cc, :] = (screen[rr, cc, :]-(val[:,None]*0.4*255).astype(np.uint8)).clip(0,255)
            #np.maximum(np.minimum(screen[rr, cc, :]-20,100)*(1-val[:,None]),0)#(screen[rr, cc, :]*(1-val[:,None])).clip(0,255).astype(np.uint8)
    # pygame.draw.aalines(screen, thread_color, False, ni[sequence,:], 1)
    #skimage.filters.gaussian(screen,sigma=10)

def pygame_show():
    import pygame
    pygame.display.flip()

    while True:
        event = pygame.event.wait()
        if event.type == pygame.KEYDOWN or event.type == pygame.QUIT:
            pygame.quit()
            break