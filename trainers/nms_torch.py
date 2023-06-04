import numpy as np
import torch
import torch.nn.functional as F

def conv_tri_torch(image, r):
    if r <= 1:
        p = 12 / r / (r + 2) - 2
        f = np.array([[1, p, 1]]) / (2 + p)
        r = 1
    else:
        f = np.array([list(range(1, r + 1)) + [r + 1] + list(range(r, 0, -1))]) / (r + 1) ** 2
    f = torch.tensor(f, dtype=image.dtype, device=image.device)[None, None, ...]
    image = F.pad(image, (r, r, r, r), mode='constant')
    image = F.conv2d(F.conv2d(image, f, stride=1), f.permute(0, 1, 3, 2), stride=1)
    return image


def grad_torch(image):
    image_np = image.cpu().numpy()
    oy, ox = [], []
    for bi in range(image.shape[0]):
        oy_, ox_ = np.gradient(image_np[bi, 0])
        oy.append(torch.from_numpy(oy_))
        ox.append(torch.from_numpy(ox_))
    oy = torch.stack(oy, dim=0).unsqueeze(1).to(device=image.device, dtype=image.dtype)
    ox = torch.stack(ox, dim=0).unsqueeze(1).to(device=image.device, dtype=image.dtype)
    return ox, oy


def interp_torch(edge, h, w, cos_o, sin_o):
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(grid_h, grid_w)
    grid_y = grid_y.to(device=edge.device)
    grid_x = grid_x.to(device=edge.device)
    interp_res = []
    for d in [-1, 1]:
        grid_y_ = torch.clamp(grid_y + d * sin_o, 0, h)
        grid_y_ = (grid_y_ / h - 0.5) * 2
        grid_x_ = torch.clamp(grid_x + d * cos_o, 0, w)
        grid_x_ = (grid_x_ / w - 0.5) * 2
        grid = torch.stack([grid_x_, grid_y_], dim=-1)
        interp_res_ = F.grid_sample(edge, grid, mode='bilinear', align_corners=False)
        interp_res.append(interp_res_)

    return interp_res


def nms_torch(edge, ori, m, h, w):
    nms = edge.clone()
    mask = (edge != 0)
    mask = mask.to(torch.float32)
    cos_o = torch.cos(ori)
    sin_o = torch.sin(ori)
    interp_maps = interp_torch(edge, h, w, cos_o, sin_o)
    edgem = edge * m
    for interp_map in interp_maps:
        nms[edgem < interp_map] = 0
    nms = edge * (1 - mask) + nms * mask
    return nms


def get_nms(edge_pred, binary_threshold=55):
    edge_pred = conv_tri_torch(edge_pred, r=1)
    edge_pred2 = conv_tri_torch(edge_pred, r=5)
    oxt, oyt = grad_torch(edge_pred2)
    oxxt, _ = grad_torch(oxt)
    oxyt, oyyt = grad_torch(oyt)
    orit = torch.arctan(oyyt * torch.sign(-oxyt) / (oxxt + 1e-5)) % np.pi
    orit = orit.squeeze(1)
    m, h, w = 1.001, int(edge_pred.shape[2]), int(edge_pred.shape[3])
    edges_nms = nms_torch(edge_pred, orit, m, h, w)
    edges_nms = torch.round(edges_nms * 255)
    edges_nms[edges_nms > binary_threshold] = 255
    edges_nms[edges_nms <= binary_threshold] = 0
    edges_nms = edges_nms / 255.

    return edges_nms
