from typing import Tuple

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import logging


def show_image(img, title=None, scale=1, ax=None):
    figsize = (img.shape[1] * scale/100, img.shape[0] * scale/100)
    logging.info(f"Showing image ... ({' x '.join([str(d) for d in figsize])})")
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.imshow(img)
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        if title is not None:
            ax.set_title(title)
        plt.show()
    else:
        ax.imshow(img)
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        if title is not None:
            ax.set_title(title)

def load_image_and_show(path: str) -> Tuple[
    int, int, np.ndarray, np.ndarray, np.ndarray
]:
    img = cv.imread('../data/wavesheep1.jpg')
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    height, width = img.shape[:2]
    # show_image(cv.cvtColor(img_gray, cv.COLOR_GRAY2RGB))
    _, ax = plt.subplots(1, 2, figsize=(width*2/100, height/100))

    imgs = [img_rgb, cv.cvtColor(img_gray, cv.COLOR_GRAY2RGB)]
    for i, ax in enumerate(ax):
        
        show_image(imgs[i], ax=ax)

        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])

    plt.show()

    return height, width, img, img_rgb, img_gray