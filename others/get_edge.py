import cv2
import numpy as np
import matplotlib.pyplot as plt
from get_midline import get_midline
import math
from C_code import fill_hole


def CLAHE(img):
    # 创建CLAHE对象
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(4, 4))
    # 限制对比度的自适应阈值均衡化
    dst = clahe.apply(img)
    return dst


def shape2edge(img):
    img_copy = img.copy()
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            if img[i, j] == 255 and np.sum(img[i - 1:i + 2, j - 1:j + 2]) < 255 * 9:
                img_copy[i, j] = 255
            else:
                img_copy[i, j] = 0
    return img_copy


def edge(path):
    img = cv2.imread(path)
    # img = cv2.resize(img, None,fx=0.5,fy=0.5)
    # img = cv2.resize(img, (400, 300))
    s = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:, :, 1]
    b = 255 - img[:, :, 0]
    B = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[:, :, 2]
    B = np.max(np.max(B)) - B
    gray = np.array(b, dtype=np.float) * 0.1 + np.array(s, dtype=np.float) * 0.2 + np.array(B, dtype=np.float) * 0.7
    gray[gray > 255] = 255

    gray = np.array(gray, dtype=np.uint8)  # CLAHE(np.array(gray, dtype=np.uint8))
    binary = gray.copy()
    binary[binary > 37] = 255
    binary[binary <= 37] = 0
    ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    print(ret)
    np.savez('hands', gray)
    mask = erode_and_dilate(mask)
    blur = cv2.blur(mask, (3, 3))
    blur = fill_hole.fill_hole(blur)
    edge_output = shape2edge(blur)
    '''plt.figure(0)
    plt.imshow(gray)

    plt.figure(1)
    plt.imshow(mask)

    plt.figure(2)
    plt.imshow(blur)

    plt.figure(3)
    plt.imshow(edge_output)
    plt.show()'''
    return edge_output


def erode_and_dilate(img):
    # kernel = np.ones((3, 3), np.uint8)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 矩形结构
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))  # 椭圆结构
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # 十字形结构
    dilation = cv2.dilate(img, kernel)  # 膨胀
    erosion = cv2.erode(dilation, kernel)
    return erosion


def get_peak(y):
    index_max = []
    index_min = []
    win_size = len(y) // 40 + 1
    print(win_size)
    step = 1
    for i in range(win_size // 2, len(y) - win_size // 2, step):
        part = np.array(y[i - win_size // 2:i + win_size // 2 + 1])
        max_ = np.max(part)
        max_index = np.argmax(part) + i - win_size // 2
        min_ = np.min(part)
        min_index = np.argmin(part) + i - win_size // 2
        if max_ > part[0] and max_ > part[-1] and max_ - max(part[0], part[-1]) > (win_size / 10):
            if max_index in index_max:
                continue
            index_max.append(max_index)
        if min_ < part[0] and min_ < part[-1] and min(part[0], part[-1]) - min_ > (win_size / 10):
            if min_index in index_min:
                continue
            index_min.append(min_index)
    return index_max, index_min


def fit_bolck(coordinate):
    new_coordinate = [[coordinate[0][0], coordinate[0][1]]]
    for i in range(1, len(coordinate)):
        dif_x = coordinate[i][0] - new_coordinate[-1][0]
        dif_y = coordinate[i][1] - new_coordinate[-1][1]
        if (np.power(dif_x, 2) + np.power(dif_y, 2)) <= 2:
            new_coordinate.append([coordinate[i][0], coordinate[i][1]])
            continue
        else:
            if dif_x < 0 and dif_y < 0:
                new_coordinate.append([coordinate[i][0] - 1, coordinate[i][1] - 1])
            elif dif_x < 0 and dif_y > 0:
                new_coordinate.append([coordinate[i][0] - 1, coordinate[i][1] + 1])
            elif dif_x > 0 and dif_y < 0:
                new_coordinate.append([coordinate[i][0] + 1, coordinate[i][1] - 1])
            elif dif_x > 0 and dif_y > 0:
                new_coordinate.append([coordinate[i][0] + 1, coordinate[i][1] + 1])
            elif dif_x < 0 and dif_y == 0:
                new_coordinate.append([coordinate[i][0] - 1, coordinate[i][1]])
            elif dif_x > 0 and dif_y == 0:
                new_coordinate.append([coordinate[i][0] + 1, coordinate[i][1]])
            elif dif_x == 0 and dif_y < 0:
                new_coordinate.append([coordinate[i][0], coordinate[i][1] - 1])
            elif dif_x == 0 and dif_y > 0:
                new_coordinate.append([coordinate[i][0], coordinate[i][1] + 1])
    return new_coordinate


def get_dist():
    path = 'hand879.jpg'
    coordinate = []
    img = edge(path)
    img = erode_and_dilate(img)
    img_copy = img.copy()
    start = 0
    for i in range(img.shape[0]):
        if np.sum(img[i, :]) < 255 * 2:
            img[i, :] = 0
            continue
        start = i
        break
    index_x = start
    index_y = 0
    for i in range(img.shape[1]):
        if img[start, -i] > 0:
            index_y = img.shape[1] - i
            break
        else:
            continue
    coordinate.append([index_x, index_y])
    while True:
        get_new = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                if img[index_x + i, index_y + j] > 0:
                    img[index_x, index_y] = 0
                    if abs(i) == abs(j):
                        img[index_x - i, index_y] = 0
                        img[index_x, index_y - j] = 0
                    index_x += i
                    index_y += j
                    get_new = 1
                    break
            if get_new == 1:
                coordinate.append([index_x, index_y])
                break
        if get_new == 0:
            for i in range(-2, 3):
                for j in range(-2, 3):
                    if i == 0 and j == 0:
                        continue
                    if img[index_x + i, index_y + j] > 0:
                        img[index_x, index_y] = 0
                        index_x += i
                        index_y += j
                        get_new = 1
                        break
                if get_new == 1:
                    coordinate.append([index_x, index_y])
                    break
        if get_new == 0:
            break
    coordinate = np.array(fit_bolck(coordinate))
    max_index, min_index = get_peak(coordinate[:, 0])
    '''plt.figure(0)
    plt.plot([i for i in range(coordinate.shape[0])], coordinate[:, 0])
    for i in range(len(max_index)):
        plt.plot(max_index[i], coordinate[max_index[i], 0], 'x', color='r')
    for i in range(len(min_index)):
        plt.plot(min_index[i], coordinate[min_index[i], 0], 'o', color='r')'''
    hand_shape = np.zeros((img_copy.shape), dtype=np.uint8)
    for i in range(coordinate.shape[0]):
        hand_shape[coordinate[i, 0], coordinate[i, 1]] = 255
    return get_finger(hand_shape, coordinate, max_index, min_index)


def list2img(single_finger):
    finger = np.zeros((np.max(single_finger[:, 0]) + 1, np.max(single_finger[:, 1]) + 1))
    for i in range(single_finger.shape[0]):
        finger[single_finger[i, 0], single_finger[i, 1]] = 255
    return np.array(finger, dtype=np.uint8)


def add_midline(finger, mid):
    for i in range(mid.shape[1]):
        finger[int(mid[0, i]), int(mid[1, i])] = 255
    return finger


def get_finger(hands_shape, coordinate, max_index, min_index):
    fingers = []
    peaks = []
    peak_index = []
    valley_index = []
    for i in range(len(max_index)):
        peak_index.append(coordinate[max_index[i], 0])
    for i in range(len(min_index)):
        valley_index.append(coordinate[min_index[i], 0])
    while len(valley_index) > 3:
        min_ = np.argmin(np.array(valley_index))
        # 剔除大拇指-----------------------------
        if min_ == 0:
            min_index.pop(0)
            valley_index.pop(0)
            if len(max_index) > 4:
                max_index.pop(0)
                peak_index.pop(0)
        else:
            min_index.pop(-1)
            valley_index.pop(-1)
            if len(max_index) > 4:
                max_index.pop(-1)
                peak_index.pop(-1)
        # --------------------------------------------
    valley_fit = [0]
    for i in valley_index:
        valley_fit.append(i)
    valley_fit.append(0)
    plt.plot(coordinate[:, 0], coordinate[:, 1])
    print(peak_index)
    print(valley_index)
    print(coordinate[min_index[:], 1])
    plt.show()
    # plot a figure------------------------------------
    plt.figure(2)
    plt.plot([i for i in range(coordinate.shape[0])], coordinate[:, 0])
    for i in range(len(peak_index)):
        plt.plot(max_index[i], peak_index[i], 'x', color='r')
    for i in range(len(valley_index)):
        plt.plot(min_index[i], valley_index[i], 'o', color='r')
    cv2.imshow('', hands_shape)
    plt.show()

    for i in range(len(valley_fit) - 1):
        index = np.argmax(np.array(valley_fit[i:i + 2]))
        forward = -1
        if index == 0:
            forward = 1
        # 分离最左手指
        x = valley_index[i + index - 1]
        start_y = coordinate[min_index[i + index - 1], 1]
        end_y = start_y + forward * 20
        while True:
            if hands_shape[x, end_y] == 0:
                end_y = end_y + forward
            else:
                break
        start_id = min_index[i + index - 1]
        single_finger = [[x, start_y]]
        while True:
            start_id += forward
            if coordinate[start_id, 0] == x and coordinate[start_id, 1] == end_y:
                single_finger.append([coordinate[start_id, 0], coordinate[start_id, 1]])
                break
            single_finger.append([coordinate[start_id, 0], coordinate[start_id, 1]])
        '''
finger = list2img(single_finger)
fingers.append(finger)
np.savez('finger%d.npz' % i, np.array(single_finger), np.array(coordinate[max_index[i], :]))
# ------------------------------------
cv2.imshow('', finger)
# plt.show()
cv2.waitKey(0)
'''
        single_finger = np.array(single_finger)
        min_x = np.min(single_finger[:, 0])
        min_y = np.min(single_finger[:, 1])
        single_finger[:, 0] -= min_x
        single_finger[:, 1] -= min_y
        fingers.append(np.array(single_finger))
        peaks.append(np.array(coordinate[max_index[i], :] - [min_x, min_y]))
    return fingers, peaks


def extract_feature(img, mid_line, b0, b1):
    feature = []
    if b0 == 0:
        for i in range(0, len(mid_line), len(mid_line) // 10):
            start = 0
            end = 0
            for j in range(img.shape[1]):
                if img[i, j] == 255:
                    start = j
                    break
            for j in range(img.shape[1] - 1, -1, -1):
                if img[i, j] == 255:
                    end = j
                    break
        feature.append(end - start)
    else:
        k = -b0
        for i in range(len(mid_line)):
            x = i
            y = mid_line[i]
            b = x - y * k
            fL = [int(k * i + b) for i in range(img.shape[1])]
            if min(fL) < 0:
                break
        fir = i
        for i in range(fir, len(mid_line), (len(mid_line) - fir) // 10):
            x = i
            y = mid_line[i]
            b = x - y * k
            fL = [int(k * i + b) for i in range(img.shape[1])]
            fH = [math.ceil(k * i + b) for i in range(img.shape[1])]
            start = 0
            end = 0
            for j in range(int(mid_line[i]), -1, -1):
                if img[fL[j], j] == 255:
                    start = j
                    break
                elif img[fH[j], j] == 255:
                    start = j
                    break
            for j in range(int(mid_line[i]), len(fL)):
                if img[fL[j], j] == 255:
                    end = j
                    break
                elif img[fH[j], j] == 255:
                    end = j
                    break
            feature.append(end - start)
    return feature


fingers, peaks = get_dist()
for i in range(len(peaks)):
    np.savez('finger%d.npz' % i, fingers[i], peaks[i])
finger = fingers[2]
peak = peaks[2]
img = list2img(finger)
cv2.imshow('', img)
cv2.waitKey(0)
b0, b1, pre, mid_point = get_midline(finger, peak)
'''  # plt.plot(finger[:, 0], finger[:, 1])


plt.imshow(img)
plt.plot(pre, mid_point[:, 0])
plt.plot(mid_point[:, 1], mid_point[:, 0], 'g.')
plt.show()
'''
new_pre = [b0 * i + b1 for i in range(img.shape[0])]
img_2 = add_midline(img.copy(), np.array([[i for i in range(img.shape[0])], new_pre]))
k = -b0
x = 5
y = new_pre[x]
b = x - y * k
fL = [int(k * i + b) for i in range(img.shape[1])]
img_3 = add_midline(img.copy(), np.array([fL, [i for i in range(img.shape[1])]]))
cv2.imshow('1', img_2)
cv2.imshow('2', img_3)
plt.imshow(img)
plt.plot([i for i in range(img.shape[1])], fL, 'r.')
plt.show()
cv2.waitKey(0)
