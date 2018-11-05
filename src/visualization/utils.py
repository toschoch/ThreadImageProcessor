
import inspect
import logging
log = logging.getLogger(__name__)

def figure(f):
    """
    Decorator for automatic data loading with pandas and saving figure
    Args:
        f: the function that actually draws the picture

    Returns:
        wrapped function

    """
    args = inspect.getargspec(f)[0]

    def wrapper(target, **kwargs):

        # call figure function with that data
        fig = f(**kwargs)

        # save figure
        fig.savefig(target,dpi=300)

    return wrapper