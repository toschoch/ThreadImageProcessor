import numpy as np
import pygame

def p1(state, target):
    state = pygame.surfarray.pixels3d(state)
    target = pygame.surfarray.pixels3d(target)
    return np.sum(np.abs(state-target))

def p2(state, target):
    state = pygame.surfarray.pixels3d(state)
    target = pygame.surfarray.pixels3d(target)
    return np.sum((state-target)**2)

