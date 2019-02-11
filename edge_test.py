import numpy as np
from random import randint

# Fills 'empty' cells with fill value. r is the radius of the kernel
def gFill(u, v, r, fill, img):

    if u < r or v < r or u > img.shape[0]-r-1 or v > img.shape[1]-r-1:
        return fill

    return img[u, v]


# Fills empty cells with values mirrored
def gMirror(u, v, r, img):
    j = u
    k = v
    if u < r:
        c=2*(r-u)-1
        print(c)
        j = u+c
    if u > img.shape[0]-r-1:
        j = img.shape[0]-r-(u-(img.shape[0]-r-1))
    if v < r:
        c=2*(r-v)-1
        k = v+c
    if v > img.shape[1]-r-1:
        # take radius from end, take from that the distance from v
        k = (img.shape[1]-r)-(v-(img.shape[1]-r-1))

    return img[j, k]


# Fills empty cells with the nearest cell
def gNN(u, v, r, img):
    if u < r:
        u = r
    if u > img.shape[0]-r-1:
        u = img.shape[0] - r-1

    if v < r:
        v = r
    if v > img.shape[1]-r-1:
        print(u,v)
        v = img.shape[1] - r-1
        print(u,v)

    return img[u, v]

radius=3

kernel=[[1,1,1],[1,1,1],[1,1,1]]
image = np.zeros((10, 8))

# fill image with data
for i in range(radius, image.shape[0]-radius):
    for j in range(radius, image.shape[1]-radius):
        image[i, j] = randint(0, 10)

# create array for storing new values
lg_image = np.zeros((10, 8))

# for i in range(0,image.shape[0]):
#     for m in range(0,image.shape[1]):
#                 lg_image[i][m] = gFill(i, m, 2, 12, image)

# for i in range(0, image.shape[0]):
#     for m in range(0, image.shape[1]):
#         lg_image[i][m] = gMirror(i, m, radius, image)

for i in range(0, image.shape[0]):
    for m in range(0, image.shape[1]):
        lg_image[i][m] = gNN(i, m, radius, image)

for i in lg_image:
    print(i)

