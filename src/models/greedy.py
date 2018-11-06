import pygame
import logging

log = logging.getLogger(__name__)

def next_step(state, current_node, target_value, target, nodes, loss, thread_color=(0,0,0)):

    log.debug("current target: {}".format(target_value))
    next_node = None
    delta = 0
    for i in range(len(nodes)):
        if i != current_node:
            log.debug("try edge to node: {}".format(i))
            candidate = state.copy()
            pygame.draw.aaline(candidate, thread_color, nodes[current_node], nodes[i], 1)
            candidate_value = loss(candidate, target)
            log.debug("candidate target: {}".format(candidate_value))
            if candidate_value < target_value:
                next_node = i
                delta += target_value - candidate_value
                target_value = candidate_value

    if next_node is None:
        log.debug("no improvement possible through adding a edge")
        return next_node, target_value

    pygame.draw.aaline(state, thread_color, nodes[current_node], nodes[i], 1)
    log.info("best next node: {} with target: {}, delta: {}".format(next_node, target_value, delta))

    return next_node, target_value
