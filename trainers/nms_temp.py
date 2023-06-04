import os
from ctypes import *

import cv2
import numpy as np

from .impl.toolbox import conv_tri, grad2

# NOTE:
#    In NMS, `if edge < interp: out = 0`, I found that sometimes edge is very close to interp.
#    `edge = 10e-8` and `interp = 11e-8` in C, while `edge = 10e-8` and `interp = 9e-8` in python.
#    ** Such slight differences (11e-8 - 9e-8 = 2e-8) in precision **
#    ** would lead to very different results (`out = 0` in C and `out = edge` in python). **
#    Sadly, C implementation is not expected but needed :(
solver = cdll.LoadLibrary("/home/wmlce/dql_inpainting/CNN_final/src/cxx/lib/solve_csa.so")
c_float_pointer = POINTER(c_float)
solver.nms.argtypes = [c_float_pointer, c_float_pointer, c_float_pointer, c_int, c_int, c_float, c_int, c_int]


def nms_process_one_image(image, save_path=None, save=True):
    """"
    :param image: numpy array, edge, model output
    :param save_path: str, save path
    :param save: bool, if True, save .png
    :return: edge
    NOTE: in MATLAB, uint8(x) means round(x).astype(uint8) in numpy
    """

    if save and save_path is not None:
        assert os.path.splitext(save_path)[-1] == ".png"
    edge = conv_tri(image, 1)
    ox, oy = grad2(conv_tri(edge, 4))
    oxx, _ = grad2(ox)
    oxy, oyy = grad2(oy)
    ori = np.mod(np.arctan(oyy * np.sign(-oxy) / (oxx + 1e-5)), np.pi)
    out = np.zeros_like(edge)
    r, s, m, w, h = 1, 5, float(1.01), int(out.shape[1]), int(out.shape[0])
    solver.nms(out.ctypes.data_as(c_float_pointer),
               edge.ctypes.data_as(c_float_pointer),
               ori.ctypes.data_as(c_float_pointer),
               r, s, m, w, h)
    edge = np.round(out * 255).astype(np.uint8)
    if save:
        cv2.imwrite(save_path, edge)
    return edge


import torch


def get_nms(edge_pred, binary_threshold=55):
    # edge_pred:[B,1,H,W] detached
    device = edge_pred.device
    edge_np = edge_pred.cpu().numpy()

    edges_nms = []
    for i in range(edge_np.shape[0]):
        try:
            edge_nms = nms_process_one_image(edge_np[i, 0], save_path=None, save=False)
            edge_nms[edge_nms > binary_threshold] = 255
            edge_nms[edge_nms <= binary_threshold] = 0
            edge_nms = edge_nms / 255.
        except:
            edge_nms = edge_np[i, 0]
        edge_nms = torch.tensor(edge_nms, device=device, dtype=torch.float32)[None, ...]
        edges_nms.append(edge_nms)

    edges_nms = torch.stack(edges_nms, dim=0)
    return edges_nms

