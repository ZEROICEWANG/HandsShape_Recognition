import pandas as pd
import numpy as np
import os
import shutil
#from get_feature import get_shape, get_dist
import cv2
import matplotlib.pyplot as plt


def select():
    path = r'E:\Folder\hands_shape'
    file = 'HandInfo.csv'
    floder = 'palm'
    hand = 'Hands'
    floder1 = 'after'
    Data = pd.read_csv(os.path.join(path, file))
    data = Data.values
    ids = data[:, 0]
    files = data[:, 7]
    accessories = data[:, 4]
    nailPolish = data[:, 5]
    aspectOfHand = data[:, 6]
    tag = -1
    count = -1
    type = ['palmar right', 'palmar left']
    for i in range(len(ids)):
        if int(ids[i]) != tag:
            count += 1
            tag = int(ids[i])
            os.mkdir(os.path.join(path, floder, str(count)))
        if accessories[i] == 0 and (aspectOfHand[i] in type):
            shutil.copyfile(os.path.join(path, hand, files[i]), os.path.join(path, floder, str(count), files[i]))
            print(files[i])


def show():
    path = r'E:\Folder\hands_shape\palm'
    save_path = r'E:\Folder\hands_shape\compare'
    floders = os.listdir(path)
    tag = 0
    for floder in floders[100:]:
        files = os.listdir(os.path.join(path, floder))
        if not os.path.exists(os.path.join(save_path, floder)):
            os.mkdir(os.path.join(save_path, floder))
        for file in files:
            img = cv2.imread(os.path.join(path, floder, file))
            img = cv2.resize(img, (800, 600))
            img_ = get_shape(img)
            new = np.zeros((img.shape[0], img.shape[1] * 2, 3), dtype=np.uint8)
            new[:, :img.shape[1], :] = img
            new[:, img.shape[1]:, 0] = img_
            new[:, img.shape[1]:, 1] = img_
            new[:, img.shape[1]:, 2] = img_
            cv2.imwrite(os.path.join(save_path, floder, file), new)
            print(floder, ' ', file)
            # print(floder + '--' + file)
            '''plt.figure(0)
            plt.subplot(121)
            plt.imshow(img)
            plt.subplot(122)
            plt.imshow(img_)
            # plt.show()
            plt.savefig(os.path.join(save_path, floder, file))'''
            print(floder, ' ', file)
            '''tag = input(floder + '--' + file + ' :please input:')
            if tag == 'd':
                os.remove(os.path.join(path, floder, file))
                print('remove ' + file)'''


def empty():
    path = r'./feature_both'
    floders = os.listdir(path)
    tag = 0
    for floder in floders:
        files = os.listdir(os.path.join(path, floder))
        if len(files) == 0:
            os.rmdir(os.path.join(path, floder))
            print(floder, len(files))


def delete():
    path = r'E:\Folder\hands_shape\palm'
    save_path = r'E:\Folder\hands_shape\compare'
    floders = os.listdir(save_path)
    files_list = []
    for floder in floders:
        files = os.listdir(os.path.join(save_path, floder))
        for file in files:
            files_list.append(file)
    floders = os.listdir(path)
    for floder in floders:
        files = os.listdir(os.path.join(path, floder))
        for file in files:
            if not (file in files_list):
                os.remove(os.path.join(path, floder, file))
                print('remove: ', floder, ' ', file)


empty()

# show()
# delete()
