import skimage
import logging
import numpy as np
from src.visualization.draw import draw_edge

log = logging.getLogger(__name__)

def next_step(state, current_node, target_value, target, nodes, loss, thread_color=1.0):

    log.debug("current target: {}".format(target_value))
    next_node = None
    delta = 0

    for i in range(len(nodes)):
        if i != current_node:
            log.debug("try edge to node: {}".format(i))
            candidate = state.copy()
            draw_edge(candidate, current_node, i, nodes, thread_color)

            candidate_value = loss(candidate, target)
            log.debug("candidate target: {}".format(candidate_value))
            if candidate_value < target_value:
                next_node = i
                delta += target_value - candidate_value
                target_value = candidate_value

    if next_node is None:
        log.debug("no improvement possible through adding a edge")
        return next_node, target_value

    draw_edge(state, current_node, next_node, nodes, thread_color)
    log.info("best next node: {} with target: {}, delta: {}".format(next_node, target_value, delta))

    return next_node, target_value
