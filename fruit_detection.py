# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import cv2
import matplotlib.pyplot as plt
import cvlib as cv
from glob import glob
from cvlib.object_detection import draw_bbox

img_names = glob('dataset/*.jpg')
images = [cv2.imread(img) for img in img_names]

for img in images:
    bbox, label, conf = cv.detect_common_objects(img)
    output_image = draw_bbox(img, bbox, label, conf)
    plt.imshow(output_image)
    plt.show()
