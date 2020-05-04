from scipy.spatial import Voronoi
import numpy as np
from Linear1 import get_regress


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


def get_midline(point, peak):
    vor = Voronoi(point)
    base = get_two_base_point(point)
    new_base = [base[0] + (base[1] - base[0]) * 0.3, base[1] - (base[1] - base[0]) * 0.3]

    mid_point = []
    vertices = vor.vertices.copy()
    for i in range(vertices.shape[0]):
        if is_inside(new_base[0][0], new_base[0][1], new_base[1][0], new_base[1][1], peak[0], peak[1],
                     vertices[i, 0], vertices[i, 1]):
            mid_point.append(vertices[i])
    mid_point = np.array(mid_point)

    b0, b1, pre, _ = get_regress(mid_point[:, 0], mid_point[:, 1])
    return b0, b1, pre, mid_point
