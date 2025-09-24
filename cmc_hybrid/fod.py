#!/usr/bin/env python

# fod.py - functions dealing with FOD-related analysis
#
# Author: Silei Zhu, Saad Jbabdi, Amy Howard
#
# Copyright (C) 2025 University of Oxford
# SHBASECOPYRIGHT


from cmc_hybrid import utils
import numpy as np

# Hybrid vectors
def hybrid_vecs(th_samples, ph_samples, f_samples, vecs):
    """Create hybrid vectors from 2D and 3D samples

    :param th_samples: bpx samples (array)
    :param ph_samples: bpx samples (array)
    :param f_samples: bpx samples (array)
    :param vecs: psoct samples (array)
    :return: array
    """
    #print(th_samples)
    #print(vecs)

    # 1) find plane for vecs
    V = np.linalg.svd(vecs, full_matrices=False)[-1].T  # 3x3

    # 2) project vecs and bpx onto plane
    v          = utils.pol2cart(th_samples, ph_samples)  # nx3
    v_plane    = utils.vec_normalise(v@V, 1)
    vecs_plane = utils.vec_normalise(vecs@V, 1)


    # 3) argmax of cosine angle
    a = (utils.vec_normalise(vecs_plane[:,:2],1) @ utils.vec_normalise(v_plane[:,:2],1).T)**2  # Nxn
    idx = np.argmax(a, axis=1)

    x  = vecs_plane[:,0]
    y  = vecs_plane[:,1]
    z  = v_plane[idx,2]

    alpha = np.sqrt((1-z**2) / (x**2+y**2))
    xyz   = [alpha*x,alpha*y,z]

    new_vecs = np.stack(xyz, axis=1)
    new_vecs = new_vecs@V.T
    new_vecs = utils.vec_normalise(new_vecs, 1)

    # 4) return 3D vector
    return new_vecs

# SPHERICAL HARMONICS STUFF
from scipy.special import sph_harm_y
def form_SHmat(coord,max_order=8, coord_system='polar'):
    """Form a Spherical Harmonics design matrix

    :param coord: list or array
    :param max_order: order of the SH
    :param coord_system: 'polar' or 'cart'
    :return:
    """
    assert coord_system in ['polar', 'cart']
    if coord_system == 'cart':
        pol,az = utils.cart2pol(coord)
    else:
        pol,az = coord
    mat = []
    for n in range(0,max_order+1,2): # only even order
        for m in range(-n,n+1):
            mat.append(sph_harm_y(n,m,pol,az).real)
    return np.asarray(mat).T
