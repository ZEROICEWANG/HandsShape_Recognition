import numpy as np
from PIL import Image
import cv2
from skimage import morphology, data, color
import matplotlib.pyplot as plt
from Linear1 import get_regress


def drawsque(gray, finger):
    start = finger[0]
    end = finger[-1]
    x = start[0]
    ymin = min(start[1], end[1])
    ymax = max(start[1], end[1])
    for i in range(ymin, ymax):
        gray[x, i] = 255
    return gray


def list2img(single_finger):
    finger = np.zeros((np.max(single_finger[:, 0]) + 1, np.max(single_finger[:, 1]) + 1))
    for i in range(single_finger.shape[0]):
        finger[single_finger[i, 0], single_finger[i, 1]] = 255
    return np.array(finger, dtype=np.uint8)


def neededimg(finger):
    img = list2img(finger)  # 稀疏矩阵转图像数组
    gray = drawsque(img, finger)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, 0, 255, cv2.FILLED)
    #     展示img
    #     cv2.imshow("img",img)
    #     cv2.waitKey(0)
    return img


def SkeletonEx(img):
    m, n = np.shape(img)
    print(m, n)
    image = img.copy()
    for i in range(m):
        for j in range(n):
            if img[i, j] == 255:
                image[i, j] = 1

    # 实施骨架算法
    skeleton = morphology.skeletonize(image)

    # 提取骨架点
    im = np.zeros(np.shape(skeleton))
    Skeleton_points = []
    for i in range(m):
        for j in range(n):
            if skeleton[i][j] == True:
                Skeleton_points.append(i)
                Skeleton_points.append(j)
                im[i][j] = 255
                break
            else:
                im[i][j] = 0

    # 寻找修正删除起始位置
    dele = 0
    for i in range(12, len(Skeleton_points), 2):
        if i + 11 >= len(Skeleton_points):
            print(i)
            break
        k1 = Skeleton_points[i + 1] - Skeleton_points[i + 7]
        k2 = Skeleton_points[i + 7] - Skeleton_points[i + 11]
        if abs(k1 - k2) > 2:
            dele = i / 2 + 3
            print('dele', dele)
            break
    print(dele)
    dele = int(dele)
    m, n = np.shape(skeleton)
    for i in range(dele):
        for j in range(n - 1):
            im[i][j] = 0

    # 寻找中轴线
    Skeleton = []
    for i in range(2 * dele, len(Skeleton_points), 2):
        point = []
        point.append(Skeleton_points[i])
        point.append(Skeleton_points[i + 1])
        Skeleton.append(point)
        img[Skeleton_points[i]][Skeleton_points[i + 1]] = 0

    return im, img, np.array(Skeleton)  # im:骨架矩阵, img:手指和手指骨架, Skeleton:骨架点列表


def Extract(finger):
    img = neededimg(finger)
    return SkeletonEx(img)


def get_midline_bone(finger):
    im, img, mid_point = Extract(finger)
    b0, b1, pre, _ = get_regress(mid_point[:, 0], mid_point[:, 1])
    return b0, b1, pre, mid_point


'''Data = np.load('finger.npz')
finger = Data['arr_0']
b0, b1, pre, mid_point = get_midline_bone(finger)
plt.figure(1)
plt.plot(finger[:, 0], finger[:, 1], 'r.')
plt.plot(mid_point[:, 0], mid_point[:, 1])
plt.plot([i for i in range(max(finger[:, 0]))], [b0 * i + b1 for i in range(max(finger[:, 0]))], 'g.')
plt.show()'''
