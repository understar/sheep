# coding: cp936

''' process the images
get the vector line (with coords)
'''

import numpy as np
import os
import matplotlib.pyplot as plt
import skimage.io as io
from skimage import filter, color

img = io.imread('sheep.png')
gray=color.rgb2gray(img)

# otsu
thresh = filter.threshold_otsu(gray)
binary = gray <= thresh

# labels binary image
import skimage.morphology as mp
import skimage.measure as ms
binary=mp.binary_closing(binary,mp.diamond(1))
#binary=mp.binary_closing(binary,mp.diamond(1))
labels = ms.label(binary, neighbors=8)
contours = ms.find_contours(labels, 0.8)

# Display the image and plot all contours found
with plt.xkcd():
    fig, ax = plt.subplots()
    ax.imshow(binary, interpolation='nearest', cmap=plt.cm.gray)
    
    for n, contour in enumerate(contours):
        ax.plot(contour[:, 1], contour[:, 0], linewidth=1)
    ax.annotate('Do you know?\n I\'m a cute sheep.', (0.2, 0.45), textcoords='axes fraction', size=20)
    
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()