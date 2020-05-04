import numpy as np
import os

path = 'feature'
files = os.listdir(path)
features = []
tags = []
for file in files:
    print(file)
    Data = np.load(os.path.join(path, file))
    feature = Data['arr_0']
    tag = Data['arr_1']
    features.append(feature)
    # print(tag)

base = features.pop(0)
files.pop(0)
difs = []
for i, feature in enumerate(features):
    dif = np.mean(np.mean(np.abs(base - feature)))
    print(files[i], dif)
