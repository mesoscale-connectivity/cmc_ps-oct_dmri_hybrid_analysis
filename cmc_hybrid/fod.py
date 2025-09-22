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
    v       = utils.pol2cart(th_samples, ph_samples)  # nx3
    v_plane = v@V
    vecs_plane = vecs@V

    # 3) argmax of cosine angle
    a = (vecs_plane[:,:2] @ v_plane[:,:2].T)**2  # Nxn
    #print(a)
    idx = np.argmax(a, axis=1)

    new_vecs = np.concatenate((vecs_plane[:,:2],v_plane[idx,2][:,None]), axis=1)
    new_vecs = new_vecs@V.T

    # 4) return 3D vector
    return new_vecs

# SPHERICAL HARMONICS STUFF
from scipy.special import sph_harm
def form_SHmat(coord,max_order=8,coord_system='polar'):
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
            mat.append(sph_harm(m,n,az,pol).real)
    return np.asarray(mat).T
