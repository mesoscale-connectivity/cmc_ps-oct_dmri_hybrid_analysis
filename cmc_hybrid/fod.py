#!/usr/bin/env python

# fod.py - functions dealing with FOD-related analysis
#
# Author: Silei Zhu, Saad Jbabdi, Amy Howard
#
# Copyright (C) 2025 University of Oxford
# SHBASECOPYRIGHT


from cmc_hybrid import utils
import numpy as np

# For Kernel Density Estimation
from scipy.spatial import KDTree


# Hybrid vectors
def hybrid_vecs(th_samples, ph_samples, f_samples, vecs, retardance, thr_retardance=19,
                output_type="h-dyad", weighted=False, deterministic=False):
    """Create hybrid vectors from 2D and 3D samples

    :param th_samples: bpx samples (array)
    :param ph_samples: bpx samples (array)
    :param f_samples: bpx samples (array)
    :param vecs: psoct samples (array)
    :param retardance: retardance values (array)
    :param thr_retardance: threshold for retardance
    :return: array
    """

    # 1) find plane for vecs
    V = np.linalg.svd(vecs, full_matrices=False)[-1].T  # 3x3

    # 2) project vecs and bpx onto plane
    v          = utils.pol2cart(th_samples, ph_samples)  # nx3
    v_plane    = utils.vec_normalise(v@V, 1)
    vecs_plane = utils.vec_normalise(vecs@V, 1)

    # 3) argmax of cosine angle
    a = (utils.vec_normalise(vecs_plane[:, :2], 1) @ utils.vec_normalise(v_plane[:, :2], 1).T)**2  # Nxn

    # 4) thresholded with the f
    mask = (f_samples > 0).astype(float)
    a = a * mask[None, :]

    # weighted by f_samples
    if weighted:
        a = a * f_samples[None, :]
    if deterministic:
        idx = np.argmax(a, axis=1)
    else:
        idx = sample_from(a, axis=1)

    # 5) negate the inplane to align mic and mri
    v_plane_part = v_plane[idx, :]
    cosang = np.sum(vecs_plane[:, :2] * v_plane_part[:, :2], axis=1)
    neg = cosang < 0
    vecs_plane[neg] *= -1.0

    # threshold with retardance
    if retardance is not None:

        retardance = np.array([retardance])
        mask = retardance > thr_retardance
        chosen_xy = np.where(mask.squeeze()[:, None], vecs_plane[:, :2], 0)
        x = chosen_xy[:, 0]
        y = chosen_xy[:, 1]

    else:
        x  = vecs_plane[:, 0]
        y  = vecs_plane[:, 1]

    z  = v_plane[idx, 2]

    x, y, z = x[x*y != 0], y[x*y != 0], z[x*y != 0]

    alpha = np.sqrt((1-z**2) / (x**2+y**2))

    if output_type == "inplane-dMRI":
        xyz   = [v_plane[idx, 0], v_plane[idx, 1], np.zeros_like(v_plane[idx, 2])]
    else:
        xyz   = [alpha*x, alpha*y, z]

    new_vecs = np.stack(xyz, axis=1)
    new_vecs = new_vecs@V.T
    new_vecs = utils.vec_normalise(new_vecs, 1)

    return new_vecs


def sample_from(dens, axis=0):
    """Sample from density defined by histrogram

    :param dens: array
    :param axis: axis along which the density is defined
    :return: array
    """
    assert dens.ndim == 2,    "only supports two dimensional arrays"
    assert axis < 2,          "only supports two dimensional arrays"
    assert np.min(dens) >= 0, "density must be positive"

    prob       = dens / np.sum(dens, axis=axis, keepdims=True)
    cumprob    = np.cumsum(prob, axis=axis)
    other_axis = 1-axis
    r          = np.random.uniform(size=prob.shape[other_axis])
    r          = r[:, None] if axis == 1 else r[None, :]
    p          = (cumprob > r).astype(int)
    index      = np.argmax(p, axis=axis)
    return index


# SPHERICAL HARMONICS STUFF
def form_SHmat(coord, max_order=8, coord_system='polar'):
    """Form a Spherical Harmonics design matrix

    :param coord: list or array
    :param max_order: order of the SH
    :param coord_system: 'polar' or 'cart'
    :return:
    """
    assert coord_system in ['polar', 'cart']

    from scipy.special import sph_harm_y

    if coord_system == 'cart':
        pol, az = utils.cart2pol(coord)
    else:
        pol, az = coord
    mat   = []
    sqrt2 = np.sqrt(2.)
    for n in range(0, max_order+1, 2):  # only even order
        for m in range(-n, n+1):
            if m < 0:
                mat.append(sqrt2*sph_harm_y(n, -m, pol, az).imag)
            elif m > 0:
                mat.append(sqrt2*sph_harm_y(n, m, pol, az).real)
            else:
                mat.append(sph_harm_y(n, m, pol, az).real)
    return np.array(mat).T


class SphereKernelDensity(object):
    def __init__(self, bandwidth=1., n_beighbours=10):
        self.h = bandwidth
        self.K = n_beighbours
        self.lnC3 = np.log(bandwidth / np.sinh(bandwidth)/4./np.pi)
        self.tree = None
        self.npts = None

    def fit(self, xyz):
        self.npts = xyz.shape[0]
        self.tree = KDTree(xyz)

    def pdf(self, xyz, weights=None):
        dist, idx = self.tree.query(xyz, k=self.K, p=2)
        dp   = 1-dist**2/2.
        logprob = self.h*dp + self.lnC3

        # weighted mean?
        if weights is not None:
            if len(weights) == self.npts/2:
                weights = np.concatenate([weights, weights])
            return np.sum(np.exp(logprob) * weights[idx], axis=1) / np.sum(weights)

        return np.mean(np.exp(logprob), axis=1)


def fit_sh_fod(xyz, max_order=4, symmetric=True, weights=None, kde_bw=20., normalise=False, output_kde=False):
    """Fit spherical harmonics FOD to a bunch of sample orientations

    xyz (array)      : Nx3 input cartesian vectors
    max_order (int)  : order of spherical harmonics
    symmetric (bool) : fit to polar-symmatric set of vectors (i.e. add negatives of input vectors)
    weights (array)  : weights of various sample orientations
    kde_bw (float)   : bandwidth for KDE estimation
    normalise (bool) : output normalised SH coeffs

    Returns: SH coefficients (array)

    """
    # Identify rows that contain NaN or Inf
    valid_rows = ~np.any(np.isnan(xyz) | np.isinf(xyz), axis=1)
    xyz = xyz[valid_rows]
    if xyz.shape[0] == 0:
        print("Warning: No valid rows left after removing NaN or Inf values. Using max order = 8")
        return np.zeros(45)

    if weights is not None:
        assert len(weights) == xyz.shape[0], \
            f"length of weights {len(weights)} incompatible with length of input {xyz.shape[1]}"

    # 1) Kernel Density Estimation on the sphere
    if symmetric:
        xyzxyz  = np.concatenate([xyz, -xyz], axis=0)
    else:
        xyzxyz  = xyz
    # Enforce normalisation
    xyzxyz  = utils.vec_normalise(xyzxyz, 1)
    # KDE fit
    skde   = SphereKernelDensity(bandwidth=kde_bw)
    skde.fit(xyzxyz)
    # 2) Use grid to fit SH
    th, ph  = np.mgrid[0:2*np.pi:100j, 0:2*np.pi:100j]
    bvecs   = utils.pol2cart(th, ph).reshape((-1, 3))
    dens    = skde.pdf(bvecs, weights=weights)
    SHmat   = form_SHmat(bvecs, max_order=max_order, coord_system='cart')
    coeff   = np.linalg.pinv(SHmat)@dens
    if normalise:
        coeff = coeff / coeff[0] / np.sqrt(4.*np.pi)
    if output_kde:
        return coeff, skde
    return coeff


# ================= PLOTTING TOOLS ==================== #


LM_dict = {
    0:  1,
    2:  6,
    4:  15,
    6:  28,
    8:  45,
    10: 66,
    12: 91,
}
ML_dict = res = dict((v, k) for k, v in LM_dict.items())


def SHorder(M):
    max_M = 91
    if M > max_M:
        raise Exception("M should be <=91")
    return ML_dict[M]


def SHcoeffLen(L):
    if L <= 12:
        if np.remainder(L, 2):
            return LM_dict[L-1]
        else:
            return LM_dict[L]
    else:
        L_over_2 = int(np.floor(L/2))
        return int(L_over_2+1 + 2*L_over_2*(L_over_2+1))


def plot_odf_glyph(coeff, glyph=False, samples=None, notebook=False):
    """
    Plot signal ODF glpyh

    Args:
      coeff (array)   : SH coefficients
      glyph (bool)    : plot as deformed surface
      samples (array) : orientation samples to be scatter plotted
      notebook (bool) : plot within a jupyter notebook

    Returns:
        Plotly figure
    """
    import plotly.graph_objects as go

    if notebook:
        from plotly.offline import init_notebook_mode
        init_notebook_mode(connected=True)

    th, ph = np.mgrid[0:2*np.pi:100j, 0:2*np.pi:100j]

    bvecs = utils.pol2cart(th, ph)
    X, Y, Z = bvecs[:, :, 0], bvecs[:, :, 1], bvecs[:, :, 2]

    if isinstance(coeff, np.ndarray):
        max_order = SHorder(len(coeff))
        SHmat = form_SHmat(bvecs.reshape((-1, 3)), max_order=max_order, coord_system='cart')
        FOD   = (SHmat@coeff).reshape(th.shape)
    else:
        FOD = coeff.pdf(bvecs.reshape((-1, 3))).reshape(th.shape)

    data = []
    if glyph:
        M = np.max(FOD)
        X, Y, Z = FOD*X/M, FOD*Y/M, FOD*Z/M
    data.append(go.Surface(x=X, y=Y, z=Z, surfacecolor=FOD, showscale=False))

    if samples is not None:
        data.append(go.Scatter3d(x=samples[:, 0], y=samples[:, 1], z=samples[:, 2],
                                 mode='markers', marker=dict(size=3, color='#000000', opacity=0.8)))

    fig = go.Figure()
    fig = go.Figure(data=data)
    fig.update_layout(scene=dict(aspectmode="data"))
    # fig.show()
    return fig


# 2D Plotting (copied from SPOT)
def plot_FOD(bins, counts, ax=None):
    """ Polar plot 2D FOD

    :param bins: 1D array
    :param counts: 1D array
    :param ax: axis for plotting
    :return: None
    """
    from matplotlib import pyplot as plt
    if ax is None:
        ax = plt.subplot(111, projection='polar')
    if isinstance(counts, list):
        for c in counts:
            ax.plot(bins, c)
    else:
        ax.plot(bins, counts)
    ax.set_yticks([]), ax.set_xticks([])


def plot_FOD_from_samples(A, nbins=101, normalise=False, ax=None):
    """ Plot FOD in polar

    :param A: (N,2) array
    :param nbins: int
    :param ax: axis to plot in
    :return: None
    """
    counts, bins = FOD_from_samples(A, nbins)
    if normalise:
        counts = counts/np.sum(counts)
    plot_FOD(bins, counts, ax)


# ------------ MISC ------------------- #
def FOD_from_samples(A, nbins):
    """Histogram (FOD) of orientations in A (2-D).

    :param A 2D array (N,2)
    :param nbins (int)

    :returns counts (1D array), angle bins (1D array)
    """
    if A.ndim > 2:
        A = A.reshape(-1, A.shape[-1])
    thetas = np.linspace(-np.pi, np.pi, nbins)
    counts = np.zeros_like(thetas)
    A0 = np.stack((np.cos(thetas), np.sin(thetas)), axis=1)

    from scipy.spatial import KDTree
    tree = KDTree(A0)
    _, ii = tree.query(A, k=1)
    counts += np.bincount(ii, minlength=len(counts))
    _, ii = tree.query(-A, k=1)              # antipodal
    counts += np.bincount(ii, minlength=len(counts))
    return counts, thetas
