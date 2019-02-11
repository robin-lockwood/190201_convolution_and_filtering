import numpy as np
from random import randint


# Fills 'empty' cells with fill value and r is the radius of the kernel
def g_fill(u, v, r, fill, img):

    if u < r or v < r or u > img.shape[0]-r-1 or v > img.shape[1]-r-1:
        return fill

    return img[u, v]


# Fills empty cells with values mirrored
def g_mirror(u, v, r, img):
    j = u
    k = v
    if u < r:
        # the distance is doubled from the r-axis
        c = 2*(r-u)-1
        j = u+c
    if u > img.shape[0]-r-1:
        j = (img.shape[0]-r)-(u-(img.shape[0]-r-1))
    if v < r:
        c = 2*(r-v)-1
        k = v+c
    if v > img.shape[1]-r-1:
        # take radius from end, take from that the distance from v
        k = (img.shape[1]-r)-(v-(img.shape[1]-r-1))

    return img[j, k]


# Fills empty cells with the nearest cell
def g_nn(u, v, r, img):
    if u < r:
        u = r
    if u > img.shape[0]-r-1:
        u = img.shape[0] - r-1

    if v < r:
        v = r
    if v > img.shape[1]-r-1:
        v = img.shape[1] - r-1

    return img[u, v]


# Corresponds to kernel radius
radius = 4

# This image has extra rows and columns
image = np.zeros((20, 16))

# fill image with data
for n in range(radius, image.shape[0]-radius):
    for o in range(radius, image.shape[1]-radius):
        image[n, o] = randint(0, 10)

# create array for storing new values
lg_image = np.zeros((20, 16))

#################################
# Test these with different radii
###################################
# for i in range(0, image.shape[0]):
#     for m in range(0, image.shape[1]):
#                 lg_image[i][m] = g_fill(i, m, 2, 12, image)
#
for i in range(0, image.shape[0]):
    for m in range(0, image.shape[1]):
        lg_image[i][m] = g_mirror(i, m, radius, image)
#
# for i in range(0, image.shape[0]):
#     for m in range(0, image.shape[1]):
#         lg_image[i][m] = g_nn(i, m, radius, image)

# print to verify
for i in lg_image:
    print(i)
