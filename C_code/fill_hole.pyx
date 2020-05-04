import cv2
import numpy as np


def fill_hole(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] > 0:
                px = i
                py = j
                break
    img_dst = np.zeros(img.shape, dtype=np.uint8)
    se = np.array([[-1, 1, -1], [1, 1, 1], [-1, 1, -1]])
    tempImg = np.ones(img.shape, dtype=np.uint8) * 255
    revImg = tempImg - img
    img_dst[px, py] = 255
    while True:
        temp = img_dst.copy()
        img_dst = cv2.dilate(img_dst, se)
        img_dst = cv2.bitwise_and(img_dst, revImg)
        if np.max(np.max(img_dst - temp)) == 0:
            break
    return 255-img_dst