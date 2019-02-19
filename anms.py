import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from math import exp
from math import sqrt


# Read image
# img_color = plt.imread('noisy_big_chief.jpeg')
# img_color = plt.imread('image_1.jpg')
# I_gray = img_color.mean(axis=2)
#
# # Output original grayscale image
# fig = plt.figure(figsize=(18, 16), dpi=80, facecolor='w', edgecolor='k')
# ps = plt.imshow(I_gray, cmap='gray')
# plt.show()


# There are more elegant ways we could do this - but this works
# g(u,v) represents the sum of all pixels around u,v
def gdef(u, v, img):
    sum = 0.0
    sum += img[u - 1, v - 1] + img[u - 1, v] + img[u - 1, v + 1]
    sum += img[u, v - 1] + img[u, v] + img[u, v + 1]
    sum += img[u + 1, v - 1] + img[u + 1, v] + img[u + 1, v + 1]

    return sum


# h(u,v) represents any h(j,k)
def hdef(u, v, img, sigma):
    return 1.0 / 9.0


def convolve(g, h, img):
    out = np.zeros((img.shape[0] - 2, img.shape[1] - 2))  # Output image has 2 less rows and columns from convolution

    # Simply loop through all pixels in the image and apply g and h to those pixels
    for u in range(1, out.shape[0] + 1):
        for v in range(1, out.shape[1] + 1):
            out[u - 1, v - 1] = g(u, v, img) * h(u, v, img, 0.0)

    return out


# Our h gauss function
def hgauss(j, k, img, rad, sigma):
    val = j * j + k * k
    val = val / (2 * sigma * sigma)
    val = exp(-val)

    return val * 1 / ((2 * rad + 1) ^ 2)


# A generic G function
def gn(u, v, img):
    return img[u, v]
    return 0.0


# A generic convolution function
def convolveGauss(g, h, img, rad, sigma):
    out = np.zeros((img.shape[0] - rad - rad,
                    img.shape[1] - rad - rad))  # Output image has 2 less rows and columns from convolution

    ja = list(range(-rad, rad + 1))
    ka = list(range(-rad, rad + 1))

    # Simply loop through all pixels in the image and apply g and h to those pixels
    for u in range(rad, img.shape[0] - rad - rad):
        for v in range(rad, img.shape[1] - rad - rad):
            sum = 0.0

            for j in ja:
                for k in ka:
                    sum += g(u + j, v + k, img) * h(j, k, img, rad, sigma)

            out[u - rad, v - rad] = sum

    return out


def hsobel(j, k, img, rad, sigma):
    matr = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    j = j + 1
    k = k + 1
    return matr[j, k]


def hsobelT(j, k, img, rad, sigma):
    matr = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    matr = np.transpose(matr)
    j = j + 1
    k = k + 1
    return matr[j, k]


def Harris(img, radius, sd):
    Iu = convolveGauss(gn, hsobel, img, 1, 1)
    Iv = convolveGauss(gn, hsobelT, img, 1, 1)

    Iuu = np.multiply(Iu, Iu)
    Ivv = np.multiply(Iv, Iv)
    Iuv = np.multiply(Iu, Iv)

    Iuu = convolveGauss(gn, hgauss, Iuu, radius, sd)
    Ivv = convolveGauss(gn, hgauss, Ivv, radius, sd)
    Iuv = convolveGauss(gn, hgauss, Iuv, radius, sd)

    return np.multiply(Iuu, Ivv) - np.multiply(Iuv, Iuv)


def get_high_intensity(img):
    p = []
    avg = np.average(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] > avg:
                p.append(((i, j), img[i, j]))
    return p


# function to return 2-D distance
def distance(p1, p2):
    return sqrt(pow((p1[0] - p2[0]), 2) + pow((p1[1] - p2[1]), 2))


# adaptive non-maximal suppression
# points describes a matrix of tuples for storing (x,y) and intensity 
# n is the number of top radii to pick from
# diameter filters radii that are too close together
# t is a scaling factor for what intensity is considered to be. i.e. t=1 no scale; t=0.9 intensity a little less; t=1.1 intensity a little more.
def anms(points, n=500, diameter=5, t=0.9):
    s = []
    # step by d
    for i in range(len(points)):
        radius = np.Infinity
        p = points[i]
        # for every point find the closest higher intensity
        for j in range(len(points)):
            pp = points[j]
            if (i != j) and (p[1] < t * pp[1]):
                d = distance(p[0], pp[0])
                if d < radius:
                    radius = d
		# only record if distance is far enough away
        if radius > diameter:
            s.append(((p[0][0], p[0][1]), radius))
	# sort by radius
    s.sort(key=lambda l: l[1], reverse=True)

    # return a matrix of the top n points
    return s[0:n]


img_color = plt.imread('chessboard.png')
# I_gray = img_color.mean(axis=2)
cimg = Harris(img_color, 1, 2.0)

fig = plt.figure(figsize=(18, 16), dpi=80, facecolor='w', edgecolor='k')
ps = plt.imshow(cimg, cmap='gray')
plt.show()

hih_points = get_high_intensity(cimg)
descriptors = anms(hih_points, n=100, diameter=10, t=1)
filtered_image = np.zeros(cimg.shape)

for x in range(cimg.shape[0]):
    for y in range(cimg.shape[1]):
        for z in descriptors:
            if z[0][0] == x and z[0][1] == y:
                filtered_image[x, y] = 1

fig = plt.figure(figsize=(18, 16), dpi=80, facecolor='w', edgecolor='k')
ps = plt.imshow(filtered_image, cmap='gray')
plt.show()
