from PIL import Image
import PIL.ImageFilter
import numpy
import colorsys
import scipy.ndimage
import matplotlib.pyplot as plt


def hue_slicing(I, threshold):
    J = numpy.array(I)
    nrow, ncol, _ = J.shape
    for i in range(nrow):
        for j in range(ncol):
            R, G, B = J[i][j][0], J[i][j][1], J[i][j][2]
            h, _, _ = colorsys.rgb_to_hsv(R, G, B)
            h *= 360
            if abs(h - 15) > threshold:
                J[i][j] = numpy.array([0, 0, 0])
    return Image.fromarray(J)


def thresholding(I, threshold):
    i = numpy.asarray(I)
    i = i > threshold  # boolean matrix
    i = i * 255
    return Image.fromarray(i)


def erosion(I, se):
    i = scipy.ndimage.binary_erosion(Ibn, se) * 255
    return Image.fromarray(i)


def fill_holes(I):
    i = scipy.ndimage.binary_fill_holes(Ibn) * 255
    return Image.fromarray(i)


def create_mask(I, threshold):
    J = numpy.array(I)
    nrow, ncol = J.shape
    threshold *= 255
    mask = {'rows': [], 'cols': []}
    for i in range(nrow):
        if numpy.sum(J[i, :]) > threshold:
            mask['rows'].append(i)
            break
    for i in range(nrow - 1, -1, -1):
        if numpy.sum(J[i, :]) > threshold:
            mask['rows'].append(i)
            break
    for i in range(ncol):
        if numpy.sum(J[:, i]) > threshold:
            mask['cols'].append(i)
            break
    for i in range(ncol - 1, -1, -1):
        if numpy.sum(J[:, i]) > threshold:
            mask['cols'].append(i)
            break
    return mask


def apply_mask(I, mask):
    J = numpy.array(I)
    nrow, ncol, _ = J.shape
    for row in mask['rows']:
        for i in range(mask['cols'][0], mask['cols'][1]):
            J[row, i] = numpy.array([0, 0, 0])
    for col in mask['cols']:
        for i in range(mask['rows'][0], mask['rows'][1]):
            J[i, col] = numpy.array([0, 0, 0])
    return Image.fromarray(J)


path = 'sample8.jpg'
result_path = 'r8.jpg'
hue_threshold = 10
binary_threshold = 50
median_filter_size = 15
erosion_se = [
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0]
]
mask_threshold = 20

print("Loading the Image...")
Iraw = Image.open(path)
print("Slicing by Hue...")
Islc = hue_slicing(Iraw, hue_threshold)
print("Filtering with Median Filter...")
Imdf = Islc.filter(PIL.ImageFilter.MedianFilter(median_filter_size))
print("Converting to Grayscale...")
Igs = Imdf.convert(mode="L")  # rgb2gray
print("Converting to Binary...")
Ibn = thresholding(Igs, binary_threshold)
print("Eroding...")
Ier = erosion(Ibn, erosion_se)
print("Filling the Holes using Morphological Operations...")
Ifld = fill_holes(Ier)
print("Creating the Rectangle Mask...")
mask = create_mask(Ifld, mask_threshold)
print("Applying the Mask...")
Ires = apply_mask(Iraw, mask)
Ires.save(result_path)

plt.subplot(2, 4, 1)
plt.title("Original")
plt.axis('off')
plt.imshow(Iraw)
plt.subplot(2, 4, 2)
plt.title("Sliced")
plt.axis('off')
plt.imshow(Islc)
plt.subplot(2, 4, 3)
plt.title("Median Filter")
plt.axis('off')
plt.imshow(Imdf)
plt.subplot(2, 4, 4)
plt.title("Grayscale")
plt.axis('off')
plt.imshow(Igs, cmap='gray')
plt.subplot(2, 4, 5)
plt.title("Binary")
plt.axis('off')
plt.imshow(Ibn)
plt.subplot(2, 4, 6)
plt.title("Erosion")
plt.axis('off')
plt.imshow(Ier)
plt.subplot(2, 4, 7)
plt.title("Hole Filling")
plt.axis('off')
plt.imshow(Ifld)
plt.subplot(2, 4, 8)
plt.title("Result")
plt.axis('off')
plt.imshow(Ires)
Ires.show()
plt.show()

