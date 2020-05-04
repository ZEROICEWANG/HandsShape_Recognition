from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import numpy as np
from Linear1 import get_regress


##验证三角内部定位算法的正确性

def get_two_base_point(points):
    left = 0
    right = 0
    '''for i in range(len(points)):
        if abs(points[i, 1] - points[i + 5, 1]) < 5:
            left = points[i, :]
            break
    for i in range(len(points) - 1, -1, -1):
        if abs(points[i, 1] - points[i - 5, 1]) < 5:
            right = points[i, :]
            break'''
    left = points[0, :]
    right = points[-1, :]
    return [left, right]


def IsTrangleOrArea(x1, y1, x2, y2, x3, y3):
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)


def is_inside(x1, y1, x2, y2, x3, y3, x, y):
    v1 = np.array([x1 - x, y1 - y, 0])
    v2 = np.array([x2 - x, y2 - y, 0])
    v3 = np.array([x3 - x, y3 - y, 0])
    p1 = v1[0] * v2[1] - v1[1] * v2[0]
    p2 = v2[0] * v3[1] - v2[1] * v3[0]
    p3 = v3[0] * v1[1] - v3[1] * v1[0]
    if p1 * p2 > 0 and p2 * p3 > 0:
        return True
    else:
        return False


points = np.load('finger0.npz')
point = points['arr_0']#手指轮廓的二值矩阵
peak = points['arr_1']#手指指尖的坐标
vor = Voronoi(point)
fig = voronoi_plot_2d(vor)

plt.figure(2)
plt.plot(point[:, 0], point[:, 1], 'g.')
plt.plot(peak[0], peak[1], 'r.')

base = get_two_base_point(point)
plt.plot(base[0][0], base[0][1], 'b.')
plt.plot(base[1][0], base[1][1], 'b.')

new_base = [base[0] + (base[1] - base[0]) * 0.3]
new_base.append(base[1] - (base[1] - base[0]) * 0.3)

plt.figure(3)
plt.plot(vor.vertices[:, 0], vor.vertices[:, 1], 'r.')
index = np.zeros((4, 2))
index[0, 0] = new_base[0][0]
index[0, 1] = new_base[0][1]
index[1, 0] = new_base[1][0]
index[1, 1] = new_base[1][1]
index[2, 0] = peak[0]
index[2, 1] = peak[1]
index[3, 0] = new_base[0][0]
index[3, 1] = new_base[0][1]
plt.plot(index[:, 0], index[:, 1], 'b')

mid_point = []
vertices = vor.vertices.copy()
for i in range(vertices.shape[0]):
    if is_inside(new_base[0][0], new_base[0][1], new_base[1][0], new_base[1][1], peak[0], peak[1],
                 vertices[i, 0], vertices[i, 1]):
        mid_point.append(vertices[i])
mid_point = np.array(mid_point)

plt.figure(4)
plt.plot(point[:, 0], point[:, 1], 'y.')
plt.plot(mid_point[:, 0], mid_point[:, 1], 'g.')
plt.plot(new_base[0][0], new_base[0][1], 'b.')
plt.plot(new_base[1][0], new_base[1][1], 'b.')
plt.plot(peak[0], peak[1], 'r.')
_, _, pre, _ = get_regress(mid_point[:, 0], mid_point[:, 1])
plt.plot(mid_point[:, 0], pre, 'r')
plt.show()
