from typing import Dict, Any
import utils
import numpy as np

import time


NDArray = Any


def calc_c_mat(img):
    gs = utils.to_grayscale(img)

    left = np.roll(gs, shift=1, axis=1)
    left[:, 0] = 0
    right = np.roll(gs, shift = -1, axis=1)
    right[:, -1] = 0
    up = np.roll(gs, shift = 1, axis=0)
    up[0, :] = 0
    left_minus_right = np.abs(left-right)
    left_minus_right[:, 0] = 225.0
    left_minus_right[:, -1] = 225.0
    CL = left_minus_right + np.abs(up-left)
    CV = left_minus_right
    CR = left_minus_right + np.abs(up-right)

    return CL, CV, CR


def calc_cost(img, forward_implementation):
    """
    calculate the cost matrix and backtrack matrix
    """
    h, w, _ = img.shape

    E = utils.get_gradients(img)
    M = np.copy(E)
    backtrack = np.zeros_like(E, dtype=int)

    if forward_implementation:
        CL, CV, CR = calc_c_mat(img)
    else:
        CL, CV, CR = 0, 0, 0

    for i in range(1,h):
        M_i_left = np.add(np.roll(M[i-1], shift=1), M[i]) + CL
        M_i_left[0] = float("inf")
        M_i_mid = np.add(M[i-1], M[i]) + CV
        M_i_right = np.add(np.roll(M[i-1], shift=-1), M[i]) + CR
        M_i_right[len(M_i_right)-1] = float("inf")
        left_mid_right = np.array(M_i_left)
        left_mid_right = np.vstack((left_mid_right, M_i_mid))

        left_mid_right = np.vstack((left_mid_right, M_i_right))
        M[i] = left_mid_right.min(axis=0)
        backtrack[i] = np.subtract(np.add(left_mid_right.argmin(axis=0), np.arange(len(M[i]))), np.ones(len(M[i])))

    backtrack[0] = backtrack[1]

    return M, backtrack


""" s_trace - indices in original image
    seam - indices in current version of image
"""
def find_seam(M, bt, indices):
    """
    find the minimum seam in the current state of an image and the original indices
    """
    h, w = M.shape

    seam = [0 for x in range(h)]
    s_trace = [0 for x in range(h)]

    seam[-1] = np.argmin(M[-1])
    s_trace[-1] = indices[-1, seam[-1]]

    for i in range(h-2, -1, -1):
        seam[i] = bt[i, seam[i+1]]
        s_trace[i] = indices[i, seam[i]]

    return seam, s_trace


def remove_seam(img, indices, seam):
    """
    remove a seam from an image
    """
    h, w, _ = img.shape

    for i in range(h):
        img[i] = np.roll(img[i], -seam[i], axis=0)
        indices[i] = np.roll(indices[i], -seam[i], axis=0)

    img = np.delete(img, 0, 1)
    indices = np.delete(indices, 0, 1)

    for i in range(h):
        img[i] = np.roll(img[i], seam[i], axis=0)
        indices[i] = np.roll(indices[i], seam[i], axis=0)

    return img, indices


# review!
def duplicate_seam(img, seam):
    h, w, _ = img.shape

    expanded = np.zeros((h, w+1, 3)) #.astype(dtype=int)

    for i in range(h):
        j = seam[i]

        if j == 0:
            expanded[i, j, :] = img[i, j, :]
            expanded[i, j+1:, :] = img[i, j:, :]
            # expanded[i, j+1, :] = img[i, j, :]
        elif j == w-1:
            expanded[i, :j+1, :] = img[i, :j+1, :]
            expanded[i, j+1, :] = img[i, j, :]
        else:
            expanded[i, :j, :] = img[i, :j, :]
            expanded[i, j+1:, :] = img[i, j:, :]
            expanded[i, j, :] = img[i, j, :]

    return expanded


def carve(img, indices, forward_implementation):
    """
    carve a seam out of an image
    """
    M, bt = calc_cost(img, forward_implementation)
    seam, s_trace = find_seam(M, bt, indices)
    img, indices = remove_seam(img, indices, seam)

    return img, seam, s_trace, indices


def colorize(img, color, seams):
    """
    mark seams carved in img with a given color
    """
    for seam in seams:
        for i in range(len(seam)):
            img[i, seam[i]] = color

    return img


def resize(image: NDArray, out_height: int, out_width: int, forward_implementation: bool) -> Dict[str, NDArray]:
    """

    :param image: ِnp.array which represents an image.
    :param out_height: the resized image height
    :param out_width: the resized image width
    :param forward_implementation: a boolean flag that indicates whether forward or basic implementation is used.
                                    if forward_implementation is true then the forward-looking energy is used otherwise
                                    the basic implementation is used.
    :return: A dictionary with three elements, {'resized' : img1, 'vertical_seams' : img2 ,'horizontal_seams' : img3},
            where img1 is the resized image and img2/img3 are the visualization images
            (where the chosen seams are colored red and black for vertical and horizontal seams, respecitvely).
    """
    start = time.time()

    img = image.copy()
    vert = image.copy()
    gs_image = utils.to_grayscale(image)
    in_height, in_width = gs_image.shape
    x_diff, y_diff = out_width-in_width, out_height-in_height
    indices = np.indices(gs_image.shape)[1]
    v_seams, h_seams = None, None
    seams = None
    red = (255, 0, 0)
    black = (0, 0, 0)

    for i in range(abs(x_diff)):
        img, seam, v_trace, indices = carve(img, indices, forward_implementation)

        if i == 0:
            v_seams = np.array([v_trace])
            seams = np.array([seam])
        else:
            v_seams = np.vstack((v_seams, v_trace))
            seams = np.vstack((seams, seam))

    if x_diff > 0:
        img = image.copy()
        for i in range(len(v_seams)-1, -1, -1):
            img = duplicate_seam(img, seams[i])

    x_panded = img.copy()
    x_panded = np.rot90(x_panded)

    if v_seams is not None:
        vert = colorize(vert, red, v_seams)

    # Horizontal seams
    img = np.rot90(img)
    gs_image = utils.to_grayscale(img)
    indices = np.indices(gs_image.shape)[1]
    seams = None

    for i in range(abs(y_diff)):
        img, seam, h_trace, indices = carve(img, indices, forward_implementation)

        if i == 0:
            h_seams = np.array([h_trace])
            seams = np.array([seam])
        else:
            h_seams = np.vstack((h_seams, h_trace))
            seams = np.vstack((seams, seam))

    if y_diff > 0:
        img = x_panded.copy()
        for i in range(len(h_seams)-1, -1, -1):
            img = duplicate_seam(img, seams[i])

    hori = x_panded.copy()
    if h_seams is not None:
        hori = colorize(hori, black, h_seams)
        hori = np.rot90(hori, k=-1)

    img = np.rot90(img, k=-1)

    end = time.time()
    print(end-start)

    # return { 'resized' : img1, 'vertical_seams' : img2 ,'horizontal_seams' : img3}
    return {'resized':img, 'vertical_seams':vert, 'horizontal_seams':hori}
