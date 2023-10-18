from skimage.measure import regionprops, find_contours
from skimage.filters import threshold_otsu
from skimage.morphology import closing, square
from skimage.segmentation import clear_border
from skimage import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


PATH = "./dataFiles/test/0a4d7cbc-2522-4e51-968a-1a86d3b7ee19_5L.png"


img = io.imread(PATH)
# print(img)

# busca umbral global con método estadístico de Otsu
umbral = threshold_otsu(img)

# binariza la imagen
img_bn = (img > umbral)*1

# cierra pequeños huecos/cortes que pudiera tener la img de la mano
img_bn = closing(img_bn, square(3))

# remueve artefactos que pudiera tener la img en los bordes
img_lista = clear_border(img_bn)

# obtiene valores geométricos a partir de las regiones (objetos "aislados") en la img
regiones = regionprops(img_lista)

# datos de la primera región. Debería ser la única si la mano fue segmentada correctamente
region = regiones[0]

for prop in region:
    if prop not in ['convex_image', 'coords', 'filled_image', 'image']:
        print('%20s:    ' % prop, region[prop])

fig, axs = plt.subplots(1, 3, figsize=(10, 5))
axs[0].imshow(img, cmap='gray')
axs[1].imshow(img_bn, cmap='gray')

contour = find_contours(img_bn, 0.5)[0]
y, x = contour.T

axs[2].plot(x, y.max()-y)

plt.show()
