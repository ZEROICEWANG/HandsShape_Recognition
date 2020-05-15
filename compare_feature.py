import numpy as np
import os


def compare():
    path = 'feature_both'
    folders = os.listdir(path)
    features = []
    for folder in folders:
        file = os.listdir(os.path.join(path, folder))[0]
        Data = np.load(os.path.join(path, folder, file))
        feature = Data['arr_0']
        features.append(feature)

    result = np.zeros((len(features), len(features)))+1000
    difs=[]
    for i in range(len(features)):
        base = features[i]
        for j in range(i+1,len(features)):
            dif = np.mean(np.mean(np.abs(base - features[j])))
            difs.append(dif)
            print(dif)
            result[i, j] = dif
    for i in range(len(features)):
        result[i, i] = 1000
    difs=np.sort(np.array(difs))
    for i in difs:
        print(i)
    print('finish')


compare()
