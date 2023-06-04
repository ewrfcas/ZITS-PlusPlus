import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


def edges_eval_plot(algs, nms=None, cols=None):
    """
    See https://github.com/pdollar/edges/blob/master/edgesEvalPlot.m
    """

    # parse inputs
    nms = nms or []
    cols = cols or list("rgbkmr" * 100)
    cols = np.array(cols)
    if not isinstance(algs, list):
        algs = [algs]
    if not isinstance(nms, list):
        nms = [nms]
    nms = np.array(nms)

    # setup basic plot (isometric contour lines and human performance)
    plt.figure()
    ax = plt.gca()
    plt.box(True)
    plt.grid(True)
    plt.axhline(0.5, 0, 1, linewidth=2, color=[0.7, 0.7, 0.7])
    for f in np.arange(0.1, 1, 0.1):
        r = np.arange(f, 1.01, 0.01)
        p = f * r / (2 * r - f)
        plt.plot(r, p, color=[0, 1, 0])
        plt.plot(p, r, color=[0, 1, 0])
    h = plt.plot(0.7235, 0.9014, marker="o", markersize=8, color=[0, 0.5, 0],
                 markerfacecolor=[0, 0.5, 0], markeredgecolor=[0, 0.5, 0])
    plt.xticks(np.linspace(0, 1, 11))
    plt.yticks(np.linspace(0, 1, 11))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    ax.set_aspect('equal', adjustable='box')
    plt.axis([0, 1, 0, 1])

    # load results for every algorithm (pr=[T, R, P, F])
    n = len(algs)
    hs, res, prs = [None] * n, np.zeros((n, 9), dtype=np.float32), []
    for i, alg in enumerate(algs):
        a = "{}-eval".format(alg)
        pr = np.loadtxt(os.path.join(a, "eval_bdry_thr.txt"))
        pr = pr[pr[:, 1] >= 1e-3]
        _, o = np.unique(pr[:, 2], return_index=True)
        r50 = interp1d(pr[o, 2], pr[o, 1], bounds_error=False, fill_value=np.nan)(np.maximum(pr[o[0], 2], 0.5))
        res[i, :8] = np.loadtxt(os.path.join(a, "eval_bdry.txt"))
        res[i, 8] = r50
        prs.append(pr)
    prs = np.stack(prs, axis=0)

    # sort algorithms by ODS score
    o = np.argsort(res[:, 3])[::-1]
    res, prs, cols = res[o, :], prs[o], cols[o]
    if nms:
        nms = nms[o]

    # plot results for every algorithm (plot best last)
    for i in range(n - 1, -1, -1):
        hs[i] = plt.plot(prs[i, :, 1], prs[i, :, 2], linestyle="-", linewidth=3, color=cols[i])[0]
        prefix = "ODS={:.3f}, OIS={:.3f}, AP={:.3f}, R50={:.3f}".format(*res[i, [3, 6, 7, 8]])
        if nms:
            prefix += " - {}".format(nms[i])
        print(prefix)

    # show legend if nms provided (report best first)
    if not nms:
        plt.show()
        return

    nms = ["[F=.80] Human"] + ["[F={:.2f}] {}".format(res[i, 3], nms[i]) for i in range(n)]
    hs = h + hs
    plt.legend(hs, nms, loc="lower left")
    plt.show()

