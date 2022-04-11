from typing import Dict, Any
import utils
import numpy as np

NDArray = Any


def calc_c_mat(img):
    """
    Calculate CL, CV, CR for the forward implementation
    according to the formulas from the instructions
    """
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

    for i in range(1,h):
        M_i_left = np.add(np.roll(M[i-1], shift=1), M[i])

        if forward_implementation:
            M_i_left += CL[i]

        M_i_left[0] = float("inf")
        M_i_mid = np.add(M[i-1], M[i])

        if forward_implementation:
            M_i_mid += CV[i]

        M_i_right = np.add(np.roll(M[i-1], shift=-1), M[i])

        if forward_implementation:
            M_i_right += CR[i]

        M_i_right[len(M_i_right)-1] = float("inf")
        left_mid_right = np.array(M_i_left)
        left_mid_right = np.vstack((left_mid_right, M_i_mid))

        left_mid_right = np.vstack((left_mid_right, M_i_right))
        M[i] = left_mid_right.min(axis=0)
        backtrack[i] = np.subtract(np.add(left_mid_right.argmin(axis=0), np.arange(len(M[i]))), np.ones(len(M[i])))

    backtrack[0] = backtrack[1]

    return M, backtrack


def find_seam(M, bt, indices):
    """
    find the minimum seam in the current state of an image and the original indices
    s_trace - indices in original image
    seam - indices in current version of image
    """
    h, w = M.shape

    seam = [0 for x in range(h)]
    s_trace = [0 for x in range(h)]

    seam[-1] = np.argmin(M[-1])
    s_trace[-1] = indices[-1, seam[-1]]

    # find rest of seam using backtracking matrix and indices helper matrix
    for i in range(h-2, -1, -1):
        seam[i] = bt[i, seam[i+1]]
        s_trace[i] = indices[i, seam[i]]

    return seam, s_trace


def remove_seam(img, indices, seam):
    """
    remove a seam from an image
    """
    h, w, _ = img.shape

    # shift rows so seam will be 1st column
    for i in range(h):
        img[i] = np.roll(img[i], -seam[i], axis=0)
        indices[i] = np.roll(indices[i], -seam[i], axis=0)

    # remove 1st column
    img = np.delete(img, 0, 1)
    indices = np.delete(indices, 0, 1)

    # restore image
    for i in range(h):
        img[i] = np.roll(img[i], seam[i], axis=0)
        indices[i] = np.roll(indices[i], seam[i], axis=0)

    return img, indices


def duplicate_seam(img, seam, indices):
    """
    duplicate a given seam in an image, for exapnsion
    """
    h, w, _ = img.shape

    expanded = np.zeros((h, w+1, 3))

    for i in range(h):
        j = indices[i, seam[i]]

        if j == 0:
            expanded[i, j, :] = img[i, j, :]
            expanded[i, j+1:, :] = img[i, j:, :]

        elif j == w-1:
            expanded[i, :j+1, :] = img[i, :j+1, :]
            expanded[i, j+1, :] = img[i, j, :]

        else:
            expanded[i, :j, :] = img[i, :j, :]
            expanded[i, j+1:, :] = img[i, j:, :]
            expanded[i, j, :] = img[i, j, :]

    # update indices matrix to match expanded image state
    _, indices = remove_seam(img, indices, seam)
    stack_to_the_right = np.rot90(np.full((1, h), w))
    indices = np.hstack((indices, stack_to_the_right))

    return expanded, indices


def carve(img, indices, forward_implementation):
    """
    carve a minimum energy seam out of an image
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
    img = image.copy()                                          # current image state
    vert = image.copy()                                         # vertical seams track
    gs_image = utils.to_grayscale(image)
    in_height, in_width = gs_image.shape                        # original image dimensiona
    x_diff, y_diff = out_width-in_width, out_height-in_height   # amount to carve
    indices = np.indices(gs_image.shape)[1]                     # helper matrix to track original indices
    v_seams, h_seams = None, None                               # absolute seams (original image indices)
    seams = None                                                # relative seams (in current img state)
    red, black = (255, 0, 0), (0, 0, 0)

    # remove seams, and track them
    for i in range(abs(x_diff)):
        img, seam, v_trace, indices = carve(img, indices, forward_implementation)

        if i == 0:
            v_seams = np.array([v_trace])
            seams = np.array([seam])
        else:
            v_seams = np.vstack((v_seams, v_trace))
            seams = np.vstack((seams, seam))

    # if image needs to be expanded, duplicate traced seams
    if x_diff > 0:
        img = image.copy()
        indices = np.indices(gs_image.shape)[1]
        for i in range(len(v_seams)-1, -1, -1):
            img, indices = duplicate_seam(img, v_seams[i], indices)

    x_panded = img.copy()
    x_panded = np.rot90(x_panded)

    # show seams on original image
    if v_seams is not None:
        vert = colorize(vert, red, v_seams)

    # Horizontal seams
    img = np.rot90(img)
    gs_image = utils.to_grayscale(img)
    indices = np.indices(gs_image.shape)[1]
    seams = None

    # remove seams, and track them
    for i in range(abs(y_diff)):
        img, seam, h_trace, indices = carve(img, indices, forward_implementation)

        if i == 0:
            h_seams = np.array([h_trace])
            seams = np.array([seam])
        else:
            h_seams = np.vstack((h_seams, h_trace))
            seams = np.vstack((seams, seam))

    # if image needs to be expanded, duplicate traced seams
    if y_diff > 0:
        img = x_panded.copy()
        indices = np.indices(gs_image.shape)[1]
        for i in range(len(h_seams)-1, -1, -1):
            img, indices = duplicate_seam(img, h_seams[i], indices)

    # show seams on image after x-axis carving
    hori = x_panded.copy()
    if h_seams is not None:
        hori = colorize(hori, black, h_seams)

    hori = np.rot90(hori, k=-1)
    img = np.rot90(img, k=-1)

    # return { 'resized' : img1, 'vertical_seams' : img2 ,'horizontal_seams' : img3}
    return {'resized':img, 'vertical_seams':vert, 'horizontal_seams':hori}
