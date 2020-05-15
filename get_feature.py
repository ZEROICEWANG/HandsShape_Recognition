import cv2
import numpy as np
import matplotlib.pyplot as plt
from get_midline import get_midline
import math
from C_code import fill_hole
import os
from scipy import signal
from get_midline_bone import get_midline_bone

nband = 0.01


def CLAHE(img):
    # 创建CLAHE对象
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(4, 4))
    # 限制对比度的自适应阈值均衡化
    dst = clahe.apply(img)
    return dst


# 图像尺寸归一化
def normal_shape(img):
    for i in range(img.shape[0] - 1, -1, -1):
        if np.sum(img[i, :]) > 0:
            break
    img = img[:i + 1, :]
    for i in range(img.shape[1]):
        if np.sum(img[:, i]) > 0:
            break
    img = img[:, i:]
    for i in range(img.shape[1] - 1, -1, -1):
        if np.sum(img[:, i]) > 0:
            break
    img = img[:, :i + 1]
    img = cv2.resize(img, (600, 800))
    return img


def get_shape(img):
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
    mask = erode_and_dilate(mask)
    blur = cv2.blur(mask, (3, 3))
    # blur = fill_hole.fill_hole(blur)
    '''
    xgrad = cv2.Sobel(blur, cv2.CV_16SC1, 1, 0, ksize=3)
    ygrad = cv2.Sobel(blur, cv2.CV_16SC1, 0, 1, ksize=3)
    edge_output = cv2.Canny(xgrad, ygrad, 50, 150, False)
    '''
    return blur


def erode_and_dilate(img):
    # kernel = np.ones((3, 3), np.uint8)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 矩形结构
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))  # 椭圆结构
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # 十字形结构
    dilation = cv2.dilate(img, kernel)  # 膨胀
    erosion = cv2.erode(dilation, kernel)
    return erosion


def get_dist(img):
    img_copy = img.copy()
    contours = cv2.findContours(img_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hand_shape = np.zeros(img_copy.shape, dtype=np.uint8)
    length = 0
    index = 0
    for i in range(len(contours[0])):
        if len(contours[0][i]) > length:
            length = len(contours[0][i])
            index = i
    cv2.drawContours(hand_shape, contours[0], index, 255, 1)
    coordinate = np.vstack((contours[0][index][:, 0, 1], contours[0][index][:, 0, 0])).T
    return hand_shape, coordinate


def get_peak(y):
    b, a = signal.butter(8, nband, 'lowpass')  # 配置滤波器 8 表示滤波器的阶数
    y = signal.filtfilt(b, a, y)  # data为要过滤的信号
    index_max = []
    index_min = []
    win_size = len(y) // 10 + 1
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
    return index_max, index_min  # 峰值与谷值在list中的位置


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
        peak_index.append(coordinate[max_index[i], 0])  # 峰值的x坐标
    for i in range(len(min_index)):
        valley_index.append(coordinate[min_index[i], 0])  # 谷值的x坐标
    # 剔除大拇指-----------------------------
    while len(valley_index) > 3:
        min_ = np.argmin(np.array(valley_index))
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

    valley_fit = [0]  # 填充最两侧谷值
    for i in valley_index:
        valley_fit.append(i)
    valley_fit.append(0)

    '''plt.plot(coordinate[:,0],coordinate[:,1])
    plt.show()
    # plot a figure------------------------------------'''
    '''b, a = signal.butter(8, nband, 'lowpass')  # 配置滤波器 8 表示滤波器的阶数
    filtedData = signal.filtfilt(b, a, coordinate[:, 0])  # data为要过滤的信号
    plt.figure(0)
    plt.plot([i for i in range(coordinate.shape[0])], filtedData)
    plt.plot([i for i in range(coordinate.shape[0])], coordinate[:, 0])
    plt.show()'''

    '''plt.figure(2)
    plt.plot([i for i in range(coordinate.shape[0])], coordinate[:, 0])
    for i in range(len(peak_index)):
        plt.plot(max_index[i], peak_index[i], 'x', color='r')
    for i in range(len(valley_index)):
        plt.plot(min_index[i], valley_index[i], 'o', color='r')
    plt.figure(3)
    plt.imshow(hands_shape)
    plt.show()'''
    # ---------------------------------------------------
    for i in range(len(valley_fit) - 1):
        index = np.argmax(np.array(valley_fit[i:i + 2]))
        forward = -1
        if index == 0:
            forward = 1
        # 分离手指
        x = valley_index[i + index - 1]
        start_y = coordinate[min_index[i + index - 1], 1]
        end_y = start_y + forward * 30
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
        single_finger = np.array(single_finger)
        min_x = np.min(single_finger[:, 0])
        min_y = np.min(single_finger[:, 1])
        single_finger[:, 0] -= min_x
        single_finger[:, 1] -= min_y
        fingers.append(np.array(single_finger))
        peaks.append(np.array(coordinate[max_index[i], :] - [min_x, min_y]))
        for i in range(len(fingers)):
            np.savez('finger%d.npz' % i, fingers[i], peaks[i])
    return fingers, peaks


def extract_feature(img, mid_line, b0):
    feature = []
    feature_point = 30
    record_p = []
    index_re = []
    if b0 == 0:
        index = np.linspace(0, len(mid_line), feature_point + 2)
        for i in index[1:feature_point + 1]:
            i = int(round(i))
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
            fH = [math.ceil(k * i + b) for i in range(img.shape[1])]
            start = 0
            end = 0
            for j in range(int(mid_line[i]), -1, -1):
                if 0 < fL[j] < img.shape[0]:
                    if img[fL[j], j] == 255:
                        start = 1
                        break
                    elif img[fH[j], j] == 255:
                        start = 1
                        break
            for j in range(int(mid_line[i]), len(fL)):
                if 0 < fL[j] < img.shape[0]:
                    if img[fL[j], j] == 255:
                        end = 1
                        break
                    elif img[fH[j], j] == 255:
                        end = 1
                        break
            if start == 1 and end == 1:
                break
        first = i  # 记录求特征值的起点
        for i in range(len(mid_line)):
            if img[i, int(mid_line[i])] == 255 or img[i, int(mid_line[i]) + 1] == 255:
                break
        last = i  # 记录求特征值终点
        index = np.linspace(first + 1, last, feature_point + 2)
        for i in index[:feature_point]:
            i = int(round(i))
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
            record_p.append(fL[start:end])
            index_re.append([start, end])
            feature.append(end - start)
    return feature, record_p, index_re


def get_feature(img):
    try:
        img = get_shape(img)
        img = normal_shape(img)
        hand_shape, coordinate = get_dist(img)
        max_index, min_index = get_peak(coordinate[:, 0])
        fingers, peaks = get_finger(hand_shape, coordinate, max_index, min_index)
    except Exception as e:
        print('get error,', e)
        return 0, 0
    features = []
    for i in range(len(fingers)):
        finger = fingers[i]
        peak = peaks[i]
        img = list2img(finger)

        try:
            # b0, b1, pre, mid_point = get_midline(finger, peak)
            b0, b1, pre, mid_point = get_midline_bone(finger)
        except Exception as e:
            print('get midline error,', e)
            return 0, 0

        new_pre = [b0 * i + b1 for i in range(img.shape[0])]
        img_2 = add_midline(img.copy(), np.array([[i for i in range(img.shape[0])], new_pre]))
        try:
            feature, record_p, index_pre = extract_feature(img, new_pre, b0)
        except Exception as e:
            print('extract feature error,', e)
            return 0

        '''for i in range(len(record_p)):
            print(feature[i])
            f = record_p[i]
            index = index_pre[i]
            img_2 = add_midline(img_2.copy(), np.array([f, [i for i in range(index[0], index[1])]]))
        plt.figure(0)
        plt.imshow(img_2)
        plt.show()'''
        features.append(feature)
    if len(features) == 4:
        return features
    else:
        return 0


def process_features():
    path = r'E:\Folder\hands_shape\palm'
    #path = r'E:\Folder\hands_shape\after'
    floders = os.listdir(path)
    for floder in floders:
        files = os.listdir(os.path.join(path, floder))
        if not os.path.exists(os.path.join('./feature_both', floder)):
            os.mkdir(os.path.join('./feature_both', floder))
        for file in files:
            img = cv2.imread(os.path.join(path, floder, file))
            try:
                features = get_feature(img)
            except Exception as e:
                print('get feature error,', e)
                os.remove(os.path.join(path, floder, file))
                print('delete ', floder, '--', file)
                continue
            if features == 0:
                os.remove(os.path.join(path, floder, file))
                print('delete ', floder, '--', file)
                continue
            if len(features) == 4:
                np.savez(os.path.join('feature_both', floder, file.split('.')[0].split('_')[1]), np.array(features))


def test_one():
    path = 'data'
    file = '6514.jpg'
    img = cv2.imread(os.path.join(path, file))
    features = get_feature(img)
    np.savez(os.path.join('feature', file.split('.')[0]), np.array(features))


#process_features()
# test_one()
