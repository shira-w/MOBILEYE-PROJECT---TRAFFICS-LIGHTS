import numpy as np
import math


def calc_TFL_dist(prev_container, curr_container, focal, pp):
    norm_prev_pts, norm_curr_pts, R, foe, tZ = prepare_3D_data(prev_container, curr_container, focal, pp)
    if abs(tZ) < 10e-6:
        print('tz = ', tZ)
    elif norm_prev_pts.size == 0:
        print('no prev points')
    elif norm_prev_pts.size == 0:
        print('no curr points')
    else:
        curr_container.corresponding_ind, curr_container.traffic_lights_3d_location, curr_container.valid = calc_3D_data(
            norm_prev_pts, norm_curr_pts, R, foe, tZ)
    return curr_container


def prepare_3D_data(prev_container, curr_container, focal, pp):
    norm_prev_pts = normalize(prev_container.traffic_light, focal, pp)
    norm_curr_pts = normalize(curr_container.traffic_light, focal, pp)
    R, foe, tZ = decompose(np.array(curr_container.EM))
    return norm_prev_pts, norm_curr_pts, R, foe, tZ


def calc_3D_data(norm_prev_pts, norm_curr_pts, R, foe, tZ):
    norm_rot_pts = rotate(norm_prev_pts, R)
    pts_3D = []
    corresponding_ind = []
    validVec = []
    for p_curr in norm_curr_pts:
        corresponding_p_ind, corresponding_p_rot = find_corresponding_points(p_curr, norm_rot_pts, foe)
        Z = calc_dist(p_curr, corresponding_p_rot, foe, tZ)
        valid = (Z > 0)
        if not valid:
            Z = 0
        validVec.append(valid)
        P = Z * np.array([p_curr[0], p_curr[1], 1])
        pts_3D.append((P[0], P[1], P[2]))
        corresponding_ind.append(corresponding_p_ind)
    return corresponding_ind, np.array(pts_3D), validVec


def normalize(pts, focal, pp):
    # transform pixels into normalized pixels using the focal length and principle point
    normalize_points = []
    for pt in pts:
        x_normalize = (pt[0] - pp[0]) / focal
        y_normalize = (pt[1] - pp[1]) / focal
        normalize_points.append((x_normalize, y_normalize, 1))
    return np.array(normalize_points)


def unnormalize(pts, focal, pp):
    # transform normalized pixels into pixels using the focal length and principle point
    res = []
    for pt in pts:
        x_unnormalize = pt[0] * focal + pp[0]
        y_unnormalize = pt[1] * focal + pp[1]
        res.append((x_unnormalize, y_unnormalize))  # ,focal
    return np.array(res)


def decompose(EM):
    # extract R, foe and tZ from the Ego Motion
    # focal of exeption
    tZ = EM[2, 3]
    R = EM[:3, :3]
    t = EM[:3, 3]
    foe = (t[0] / t[2], t[1] / t[2], 1)
    return R, foe, tZ


def rotate(pts, R):
    # rotate the points - pts using R
    res = []
    for pt in pts:
        res.append(np.dot(R, pt))
    return res


def epipolar_line(p, foe):
    m = (foe[1] - p[1]) / (foe[0] - p[0])
    n = ((p[1] * foe[0] - foe[1] * p[0]) / (foe[0] - p[0]))
    return m, n


def distance_of_p(norm_pts, p, foe):
    e = epipolar_line(p, foe)
    return abs((e[0] * norm_pts[0] + e[1] - norm_pts[1]) / math.sqrt(e[0] ** 2 + 1))


def find_corresponding_points(p, norm_pts_rot, foe):
    # compute the epipolar line between p and foe
    # run over all norm_pts_rot and find the one closest to the epipolar line
    # return the closest point and its index
    min_d = 1000
    closet_point = ()
    index_point = 0
    for index, norm_pts in enumerate(norm_pts_rot):
        d = distance_of_p(norm_pts, p, foe)
        if d < min_d:
            min_d = d
            closet_point = norm_pts
            index_point = index

    return index_point, closet_point


def calc_dist(p_curr, p_rot, foe, tZ):
    # calculate the distance of p_curr using x_curr, x_rot, foe_x and tZ
    # calculate the distance of p_curr using y_curr, y_rot, foe_y and tZ
    # combine the two estimations and return estimated Z

    return ((tZ * (foe[0] - p_rot[0])) / (p_curr[0] - p_rot[0])) * (p_rot[0] / (p_rot[0] + p_rot[1])) + (
            (tZ * (foe[1] - p_rot[1])) / (p_curr[1] - p_rot[1])) * (p_rot[1] / (p_rot[0] + p_rot[1]))
