import glob
import multiprocessing as mp
import os
from shutil import rmtree

import cv2
import numpy as np
from scipy.interpolate import interp1d
from scipy.io import loadmat

from .bwmorph_thin import bwmorph_thin
from .correspond_pixels import correspond_pixels

eps = 2e-6


def edges_eval_img(im, gt, out="", thrs=99, max_dist=0.0075, thin=True, need_v=False, workers=1):
    """
    See https://github.com/pdollar/edges/blob/master/edgesEvalImg.m
    """
    eps = 2e-16

    if isinstance(thrs, list):
        k = len(thrs)
    elif isinstance(thrs, int):
        k = thrs
        thrs = np.linspace(1/(k + 1), 1-1/(k+1), k)
    else:
        raise NotImplementedError

    # load edges and ground truth
    if isinstance(im, str):
        edge = cv2.imread(im, cv2.IMREAD_UNCHANGED) / 255.
    else:
        edge = im
    assert edge.ndim == 2
    gt = [g.item()[1] for g in loadmat(gt)["groundTruth"][0]]  # 0: Segmentation, 1: Boundaries

    # evaluate edge result at each threshold
    cnt_sum_r_p = np.zeros((k, 4), dtype=np.int)  # cnt_r, sum_r, cnt_p, sum_r
    v = np.zeros((*edge.shape, 3, k), dtype=np.float32)

    if workers == 1:
        for k_ in range(k):
            e1 = edge >= max(eps, thrs[k_])
            if thin:
                e1 = bwmorph_thin(e1)
            match_e, match_g = np.zeros_like(edge, dtype=bool), np.zeros_like(edge, dtype=np.int)
            all_g = np.zeros_like(edge, dtype=np.int)
            for g in gt:
                match_e1, match_g1, _, _ = correspond_pixels(e1, g, max_dist)
                match_e = np.logical_or(match_e, match_e1 > 0)
                match_g = match_g + (match_g1 > 0)
                all_g += g

            # compute recall and precision
            cnt_sum_r_p[k_, :] = [np.sum(match_g), np.sum(all_g), np.count_nonzero(match_e), np.count_nonzero(e1)]

            if need_v:
                cs = np.array([[1, 0, 0], [0, 0.7, 0], [0.7, 0.8, 1]]) - 1
                fp = e1.astype(np.int) - match_e.astype(np.int)
                tp = match_e
                fn = (all_g - match_g) / len(gt)
                for g in range(3):
                    v[:, :, g, k_] = np.maximum(0, 1 + fn * cs[0, g] + tp * cs[1, g] + fp * cs[2, g])
                v[:, 1:, :, k_] = np.minimum(v[:, 1:, :, k_], v[:, :-1, :, k_])
                v[1:, :, :, k_] = np.minimum(v[1:, :, :, k_], v[:-1, :, :, k_])
    else:
        assert not need_v

        def _process_thrs_loop(_edge, _gt, _eps, _thrs, _thin, _max_dist, _indices, _queue):
            for _k in _indices:
                _e1 = _edge >= max(_eps, _thrs[_k])
                if _thin:
                    _e1 = bwmorph_thin(_e1)
                _match_e, _match_g = np.zeros_like(_edge, dtype=bool), np.zeros_like(_edge, dtype=np.int)
                _all_g = np.zeros_like(edge, dtype=np.int)
                for _g in _gt:
                    _match_e1, _match_g1, _, _ = correspond_pixels(_e1, _g, _max_dist)
                    _match_e = np.logical_or(_match_e, _match_e1 > 0)
                    _match_g = _match_g + (_match_g1 > 0)
                    _all_g += _g

                # compute recall and precision
                _cnt_sum_r_p = [np.sum(_match_g), np.sum(_all_g), np.count_nonzero(_match_e), np.count_nonzero(_e1)]
                _queue.put([_cnt_sum_r_p, _k])
        if workers == -1:
            workers = mp.cpu_count()
        workers = min(workers, k)
        queue = mp.SimpleQueue()
        split_indices = np.array_split(np.arange(k), workers)
        pool = [mp.Process(target=_process_thrs_loop,
                           args=(edge, gt, eps, thrs, thin, max_dist, split_indices[_], queue))
                for _ in range(workers)]
        [thread.start() for thread in pool]
        process_cnt_k = 0

        while process_cnt_k < k:
            process_cnt_sum_r_p, process_k = queue.get()
            cnt_sum_r_p[process_k, :] = process_cnt_sum_r_p
            process_cnt_k += 1
        [thread.join() for thread in pool]

    info = np.concatenate([thrs[:, None], cnt_sum_r_p], axis=1)
    if out:
        np.savetxt(out, info, fmt="%10g")
    return info, v


def compute_rpf(cnt_sum_r_p):
    r = cnt_sum_r_p[:, 0] / np.maximum(eps, cnt_sum_r_p[:, 1])
    p = cnt_sum_r_p[:, 2] / np.maximum(eps, cnt_sum_r_p[:, 3])
    f = 2 * p * r / np.maximum(eps, p + r)
    return r, p, f


def find_best_rpf(t, r, p):
    if len(t) == 1:
        bst_t, bst_r, bst_p = t, r, p
        bst_f = 2 * p * r / np.maximum(eps, p + r)
        return bst_r, bst_p, bst_f, bst_t
    a = np.linspace(0, 1, 100)[None, :]
    b = 1 - a
    t, r, p = t[:, None], r[:, None], p[:, None]
    rj = r[1:] @ a + r[:-1] @ b  # (len(T), len(A))
    pj = p[1:] @ a + p[:-1] @ b  # (len(T), len(A))
    tj = t[1:] @ a + t[:-1] @ b  # (len(T), len(A))
    fj = 2 * pj * rj / np.maximum(eps, pj + rj)
    k = np.argmax(fj).item()
    row, col = divmod(k, 100)
    bst_r, bst_p, bst_f, bst_t = rj[row, col], pj[row, col], fj[row, col], tj[row, col]
    return bst_r, bst_p, bst_f, bst_t


def edges_eval_dir(res_dir, gt_dir, cleanup=0, thrs=99, max_dist=0.0075, thin=True, workers=1):
    """
    See https://github.com/pdollar/edges/blob/master/edgesEvalDir.m
    """
    eval_dir = "{}-eval".format(res_dir)
    if not os.path.isdir(eval_dir):
        os.makedirs(eval_dir)
    filename = os.path.join(eval_dir, "eval_bdry.txt")

    # check if results already exist.
    if os.path.isfile(filename):
        return

    assert os.path.isdir(res_dir) and os.path.isdir(gt_dir)
    ids = [os.path.split(file)[-1] for file in glob.glob(os.path.join(gt_dir, "*.mat"))]
    for ci, i in enumerate(ids):
        i = os.path.splitext(i)[0]
        res = os.path.join(eval_dir, "{}_ev1.txt".format(i))
        if os.path.isfile(res):
            continue
        im = os.path.join(res_dir, "{}.png".format(i))
        gt = os.path.join(gt_dir, "{}.mat".format(i))
        print("{}/{} eval {}...".format(ci, len(ids), im))
        edges_eval_img(im, gt, out=res, thrs=thrs, max_dist=max_dist, thin=thin, workers=workers)

    # collect evaluation results
    cnt_sum_r_p = 0
    ois_cnt_sum_r_p = 0
    scores = np.zeros((len(ids), 5), dtype=np.float32)
    if isinstance(thrs, list):
        t = len(thrs)
    elif isinstance(thrs, int):
        t = np.linspace(1 / (thrs + 1), 1 - 1 / (thrs + 1), thrs)
    else:
        raise NotImplementedError

    for i, name in enumerate(ids):
        name = os.path.splitext(name)[0]
        res = os.path.join(eval_dir, "{}_ev1.txt".format(name))
        res = np.loadtxt(res, dtype=np.float32)
        t, res = res[:, 0], res[:, 1:]
        cnt_sum_r_p += res
        # compute OIS scores for image
        r, p, f = compute_rpf(res)
        k = f.argmax()
        ois_r1, ois_p1, ois_f1, ois_t1 = find_best_rpf(t, r, p)
        scores[i, :] = [i + 1, ois_t1, ois_r1, ois_p1, ois_f1]
        ois_cnt_sum_r_p += res[k, :]

    # compute ODS R/P/F and OIS R/P/F
    r, p, f = compute_rpf(cnt_sum_r_p)
    ods_r, ods_p, ods_f, ods_t = find_best_rpf(t, r, p)
    ois_r, ois_p, ois_f = compute_rpf(ois_cnt_sum_r_p[None, :])

    # compute AP/R50
    k = np.unique(r, return_index=True)[1][::-1]
    r, p, t, f, ap = r[k], p[k], t[k], f[k], 0
    if len(r) > 1:
        ap = interp1d(r, p, bounds_error=False, fill_value=0)(np.linspace(0, 1, 101))
        ap = np.sum(ap) / 100.0
    _, o = np.unique(p, return_index=True)
    r50 = interp1d(p[o], r[o], bounds_error=False, fill_value=np.nan)(np.maximum(p[o[0]], 0.5))

    bdry = np.array([[ods_t, ods_r, ods_p, ods_f, ois_r.item(), ois_p.item(), ois_f.item(), ap]])
    bdry_thr = np.stack([t, r, p, f], axis=0).T
    np.savetxt(os.path.join(eval_dir, "eval_bdry_img.txt"), scores.astype(np.float32), fmt="%.6f")
    np.savetxt(os.path.join(eval_dir, "eval_bdry_thr.txt"), bdry_thr.astype(np.float32), fmt="%.6f")
    np.savetxt(os.path.join(eval_dir, "eval_bdry.txt"), bdry.astype(np.float32), fmt="%.6f")

    if cleanup:
        for filename in os.listdir(eval_dir):
            if filename.endswith("_ev1.txt"):
                os.remove(os.path.join(eval_dir, filename))
        rmtree(res_dir)

