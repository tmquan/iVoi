from PIL import Image
import random
import math
import numpy as np
import shutil
import os
import cv2
import skimage.measure
import skimage.segmentation
def generate_voronoi_diagram(width, height, num_cells):
    image = Image.new("RGB", (width, height))
    putpixel = image.putpixel
    imgx, imgy = image.size
    nx = []
    ny = []
    nr = []
    ng = []
    nb = []
    for i in range(num_cells):
        nx.append(np.random.randint(imgx))
        ny.append(np.random.randint(imgy))
        nr.append(np.random.randint(256))
        ng.append(np.random.randint(256))
        nb.append(np.random.randint(256))
    for y in range(imgy):
        for x in range(imgx):
            dmin = math.hypot(imgx-1, imgy-1)
            j = -1
            for i in range(num_cells):
                d = math.hypot(nx[i]-x, ny[i]-y)
                if d < dmin:
                    dmin = d
                    j = i
            putpixel((x, y), (nr[j], ng[j], nb[j]))
    return np.array(image) 
    # image.save("VoronoiDiagram.png", "PNG")
    # image.show()
#########################################################
dst_dir = ['train', 'valid'] 
num_images = 500

SHAPE = 256
random.seed(2020)
np.random.seed(2020)

for dst in dst_dir:
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst)


    for i in range(num_images):
        image = generate_voronoi_diagram(SHAPE, SHAPE, np.random.randint(5, 12))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        filename = os.path.join(dst, 'image_'+str(i+1).zfill(5)+'.png')
        # cv2.imwrite(filename, image)

        membr = 255- skimage.segmentation.find_boundaries(image).astype(np.uint8)*255
        filename = os.path.join(dst, 'membr_'+str(i+1).zfill(5)+'.png')
        cv2.imwrite(filename, membr)

        label = skimage.measure.label(membr).astype(np.uint8)*5
        filename = os.path.join(dst, 'label_'+str(i+1).zfill(5)+'.png')
        # cv2.imwrite(filename, label)
    


