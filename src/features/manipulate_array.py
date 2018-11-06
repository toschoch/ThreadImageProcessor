import numpy as np

def extract_canvas_circle(arr):

    #arr = pygame.surfarray.pixels3d(screen)

    # largest inscribed circle
    size = arr.shape[:2]
    n = np.min(size)

    x = ((np.arange(size[0])-size[0]/2.)*2./n)[:,None]
    y = ((np.arange(size[1])-size[1]/2.)*2./n)[None,:]
    I = (x*x+y*y) > 1

    arr[:,:,0][I]=255
    arr[:,:,1][I]=255
    arr[:,:,2][I]=255


