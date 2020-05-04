import cv2
import numpy as np

'''传入一个参数cv2.THRESH_OTSU，并且把阈值thresh设为0，算法会找到最优阈值，并作为第一个返回值ret返回'''

img = cv2.imread('hand77.jpg')
img = cv2.resize(img, (400, 400))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
kernel = np.ones((3, 3), np.uint8)
erosion = cv2.erode(mask, kernel)  # 腐蚀
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 矩形结构
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 椭圆结构
# kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))  # 十字形结构
dilation = cv2.dilate(erosion, kernel)  # 膨胀
'''
# mask=255-mask
new = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=mask)
'''
cv2.imshow('ori', img)
cv2.imshow('gray', gray)
cv2.imshow('erode', erosion)
cv2.imshow('dilation', dilation)
cv2.imshow('show', mask)
cv2.imshow('dif', np.abs(mask - dilation))
# cv2.imshow('show1', new)
cv2.waitKey(0)
