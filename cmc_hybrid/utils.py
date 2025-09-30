#!/usr/bin/env python

# utils.py - helper functions and utilities
#
# Author: Silei Zhu, Saad Jbabdi, Amy Howard
#
# Copyright (C) 2025 University of Oxford
# SHBASECOPYRIGHT

import numpy as np

# Generate directions on the sphere
# def dirgen(ndir):
#     """Use FSL's GPS to generate directions
#
#     :param ndir: integer number of directions
#     :return: ndir x 3 array
#     """
#     from fsl.wrappers import gps
#     import numpy as np
#     from fsl.utils.tempdir import tempdir
#     with tempdir():
#         gps('grot', ndir, optws=True)
#         return np.loadtxt(f'grot{ndir}.txt')
#
# Generate directions on the sphere
# Drop gps - use fibonacci instead
from functools import lru_cache
@lru_cache(None)
def dirgen(samples=1):
    """
    Creates N points uniformly-ish distributed on the sphere

    Args:
        samples : int
    """
    points = np.array((samples,3))
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

    i = np.arange(samples)
    y = 1 - 2.*(i / float(samples-1))
    r = np.sqrt(1-y*y)
    t = phi * i
    x = np.cos(t) * r
    z = np.sin(t) * r

    points = np.asarray([x,y,z]).T

    return points

    

# ----- SPHERICAL COORDS ----- #
def pol2cart(th,ph):
    x = np.sin(th)*np.cos(ph)
    y = np.sin(th)*np.sin(ph)
    z = np.cos(th)
    return np.array([x,y,z]).T

def cart2pol(xyz):
    n   = np.linalg.norm(xyz,axis=1,keepdims=True)
    n[n==0] = 1
    xyz = xyz / n
    th  = np.arccos(xyz[:,2])
    st  = np.sin(th)
    idx = (st==0)
    st[idx] = 1
    ph  = np.arctan2(xyz[:,1]/st,xyz[:,0]/st)
    ph[idx] = 0
    return th,ph
# ----- SPHERICAL COORDS ----- #

# ----- VECTOR STUFF --------- #
def make_dyads(vecs):
    """calculate the dyadic vector average for a list of vectors
    len(vecs)=N, each vec is 3x1
    """
    tens = np.array([[v[0]*v[0],v[0]*v[1],v[0]*v[2],v[1]*v[0],v[1]*v[1],v[1]*v[2],v[2]*v[0],v[2]*v[1],v[2]*v[2]] for v in vecs])
    tens = np.mean(tens, axis=0)
    tens = np.reshape(tens,(3,3))
    _,V = np.linalg.eigh(tens)
    return V[:,-1]

def vec_normalise(x, axis=0):
    """Normalise vectors to unit length

    :param x: array
    :param axis: direction of normalisation
    :return: array
    """
    return x / np.linalg.norm(x, axis=axis, keepdims=True)

def prepare_mask(maskfile, roi=None, scale=1):
    """Create mask based on existing mask, additional mask, and ROI definition

    :param maskfile: str
    :param roi: tuple or list
    :param scale: int
    :return: 3D array
    """
    from fsl.data.image import Image
    mask_img = Image(maskfile)
    if roi is not None:
        from fsl.wrappers.misc import fslroi
        from fsl.wrappers import LOAD
        mask_img = Image(fslroi(mask_img, LOAD, *roi).output)
    if scale != 1:
        mask_img = upscale_image(mask_img, scale)

    return mask_img


def upscale_image(img, scale):
    """
    :param img: Image object (FSLPY)
    :param scale: scalar
    :return: Image object
    """
    from fsl.utils.image.resample import resampleToPixdims
    from fsl.data.image import Image
    newimg, xform = resampleToPixdims(img, np.array(img.pixdim)/scale, order=0)
    return Image(newimg, xform=xform, header=img.header)


# Time things
import time
class Time(object):
    def __init__(self):
        self._start = None
    def tic(self):
        self._start = time.time()
    def toc(self):
        print(f'Elapsed time : {time.time()-self._start:.4f} seconds.')


# -------- PSOCT Utils
def fudge_psoct_orientation(theta, angle=22.):
    """
    Transform PSOCT orientation
    This transformation has been determined by trial and error
    Needs to be revisited!  (e.g. should angle be = 12deg?)

    The transformation is the following:
    theta -> theta/2.
    theta -> pi - theta + angle (deg)
    if theta < 0  : theta -> theta+pi
    if theta > pi : theta -> theta-pi

    :param theta: 1D array
    :param angle: float (degrees)
    :return: 1D array
    """
    # z = np.exp(-1j*np.array(theta))
    # z *= np.exp(np.pi+1j*2*angle)
    # theta = np.angle(z)/2.

    theta = np.array(theta)/2.
    theta = np.pi - theta + angle*np.pi/180.
    theta = np.where(theta <= 0, theta + np.pi, theta)
    theta = np.where(theta > np.pi, theta - np.pi, theta)

    return theta
