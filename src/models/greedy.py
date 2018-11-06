import skimage
import logging
import numpy as np

log = logging.getLogger(__name__)

def next_step(state, current_node, target_value, target, nodes, loss, thread_color=1.0):

    log.debug("current target: {}".format(target_value))
    next_node = None
    delta = 0

    for i in range(len(nodes)):
        if i != current_node:
            log.debug("try edge to node: {}".format(i))
            candidate = state.copy()
            rr, cc, val = skimage.draw.line_aa(nodes[current_node, 0], nodes[current_node, 1],
                                               nodes[i, 0], nodes[i, 1])
            candidate[rr, cc, :] = (candidate[rr, cc, :] - (val[:, None] * thread_color * 255).astype(np.uint8)).clip(0, 255)

            candidate_value = loss(candidate, target)
            log.debug("candidate target: {}".format(candidate_value))
            if candidate_value < target_value:
                next_node = i
                delta += target_value - candidate_value
                target_value = candidate_value

    if next_node is None:
        log.debug("no improvement possible through adding a edge")
        return next_node, target_value

    rr, cc, val = skimage.draw.line_aa(nodes[current_node, 0], nodes[current_node, 1],
                                       nodes[i, 0], nodes[i, 1])
    state[rr, cc, :] = (state[rr, cc, :] - (val[:, None] * 0.4 * 255).astype(np.uint8)).clip(0, 255)
    log.info("best next node: {} with target: {}, delta: {}".format(next_node, target_value, delta))

    return next_node, target_value
