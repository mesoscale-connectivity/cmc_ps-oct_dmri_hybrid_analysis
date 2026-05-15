#!/usr/bin/env python

# utils.py - helper functions and utilities
#
# Author: Silei Zhu, Saad Jbabdi, Amy Howard
#
# Copyright (C) 2025 University of Oxford
# SHBASECOPYRIGHT

import numpy as np
from fsl.data.image import Image

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

def prepare_mask(maskfile, roi=None, resolution=[0.4,0.4,0.4], slides=None, slide_direction='coronal'):
    """Create mask based on existing mask, additional mask, and ROI definition
    WARNING!!!! If slides are provided, the code will assume that they are coronal!

    :param maskfile: str
    :param roi: tuple or list
    :param scale: int
    :param slides: list of Image objects
    :return: 3D array
    """
    mask_img = Image(maskfile)
    if roi is not None:
        from fsl.wrappers.misc import fslroi
        from fsl.wrappers import LOAD
        mask_img = Image(fslroi(mask_img, LOAD, *roi).output)
    if not np.allclose(resolution, (0.4,0.4,0.4)):
        mask_img = anisotropic_upscale_image(mask_img, resolution)

    # only keep voxels that intersect with the slides
    slides_mask = np.zeros_like(mask_img.data)
    if slides is not None:
        from fsl.utils.image.resample import resampleToReference
        for sl_img in slides:
            factor = 1
            if slide_is_too_big(sl_img):
                factor = 10
            sl_resampled = resample_slide(sl_img, slide_direction=slide_direction, factor=factor)
            sl_resampled, xform = resampleToReference(image=sl_resampled, reference=mask_img, mode='nearest', constrain=True)
            sl_resampled[sl_resampled!=0] = 1.
            sl_resampled = Image(sl_resampled, xform=xform, header=mask_img.header)
            slides_mask += np.array(sl_resampled.data, dtype=int)
    else:
        slides_mask = np.ones_like(mask_img.data)

    mask_img = Image(mask_img.data * slides_mask, header = mask_img.header)

    return mask_img

def prepare_mask_slidedeck(maskfile, roi=None, resolution=[0.4,0.4,0.4], slidedeck=None, slide_direction='coronal', matOrWarp=None):
    """Create mask based on existing mask, additional mask, and ROI definition
    WARNING!!!! If slidedeck is provided, the code will assume that it is coronal!

    :param maskfile: str
    :param roi: tuple or list
    :param scale: int
    :param slidedeck: Image object
    :return: 3D array
    """
    mask_img = Image(maskfile)
    if roi is not None:
        from fsl.wrappers.misc import fslroi
        from fsl.wrappers import LOAD
        mask_img = Image(fslroi(mask_img, LOAD, *roi).output)
    if not np.allclose(resolution, (0.4,0.4,0.4)):
        mask_img = anisotropic_upscale_image(mask_img, resolution)

    # only keep voxels that intersect with the slides
    if slidedeck is not None:
        sl_img = slidedeck
        # If a transformation matrix of warpfield is provided, then transform slidedeck to reference space
        if matOrWarp is not None:
            from cmc_hybrid.coordinate_mapping import _matOrNifti
            format, _ = _matOrNifti(matOrWarp)
            if format == 'mat':
                from fsl.wrappers import applyxfm
                sl_img = Image(applyxfm(sl_img, maskfile, matOrWarp, out=LOAD)['out'])
            elif format == 'nii':
                from fsl.wrappers.fnirt import applywarp
                sl_img = Image(applywarp(sl_img, maskfile, LOAD, warp=matOrWarp)['out'])
        factor = 1
        if slide_is_too_big(sl_img):
            factor = 10
        # TODO review if resample_slide is valid for slide_decks or need to downsample all 3 dimensions
        sl_resampled = resample_slide(sl_img, slide_direction=slide_direction, factor=factor)
        from fsl.utils.image.resample import resampleToReference
        sl_resampled, xform = resampleToReference(image=sl_resampled, reference=mask_img, mode='nearest', constrain=True)
        sl_resampled[sl_resampled!=0] = 1.
        sl_resampled = Image(sl_resampled, xform=xform, header=mask_img.header)
        slides_mask = np.array(sl_resampled.data, dtype=int)
    else:
        slides_mask = np.ones_like(mask_img.data)

    mask_img = Image(mask_img.data * slides_mask, header = mask_img.header)

    return mask_img

def slide_is_too_big(sl_img, N_MAX = 2000):
    """Assess whether a slide is too big so it needs resampling

    :param sl_img: Image object
    :param N_MAX: int
    :return: bool
    """
    N = max(sl_img.shape)
    return N > N_MAX


def resample_slide(sl_img, slide_direction='coronal', factor=1):
    """Resample a slide

    :param sl_img: Image object
    :param slide_direction: one of 'coronal', 'sagittal', or 'axial'
    :param factor: (int). E.g. factor=2 means voxels will be twice as big
    :return: Image object
    """
    from fsl.utils.image.resample import resampleToReference, resampleToPixdims

    assert slide_direction in ['coronal', 'sagittal', 'axial'], f"slide_direction must be one of 'coronal', 'sagittal', or 'axial'"
    if factor == 1:
        return sl_img
    old_pixdim = sl_img.pixdim
    if slide_direction == 'coronal':
        new_pixdim = [factor*old_pixdim[0],        old_pixdim[1], factor*old_pixdim[2]]
    elif slide_direction == 'sagittal':
        new_pixdim = [       old_pixdim[0], factor*old_pixdim[1], factor*old_pixdim[2]]
    else: #direction == 'axial'
        slide_direction = [factor*old_pixdim[0], factor*old_pixdim[1],        old_pixdim[2]]
    sl, xform = resampleToPixdims(sl_img, new_pixdim, order=0)
    return Image(sl, xform=xform, header=sl_img.header)


def upscale_image(img, target_pixdims=None):
    """
    :param img: Image object (FSLPY)
    :param scale: scalar
    :return: Image object
    """
    from fsl.utils.image.resample import resampleToPixdims
    
    if target_pixdims is not None:
        new_dim = np.array(target_pixdims)
    else:
        new_dim = np.array(img.pixdim)
    newimg, xform = resampleToPixdims(img, new_dim, order=0)
    return Image(newimg, xform=xform, header=img.header)

def anisotropic_upscale_image(img, target_pixdims=None, order=0):
    """
    Resample an FSLPY Image to anisotropic voxel sizes.

    - If `target_pixdims` is provided (tuple/list length 3), it is used directly (in mm).

    Args:
        img: FSLPY Image
        target_pixdims: (dx,dy,dz) in mm (optional). preferred for clarity
        order: interpolation order (0=nearest, 1=linear, 3=cubic). Use 0 for labels.

    Returns:
        FSLPY Image (resampled)
    """
    from fsl.utils.image.resample import resampleToPixdims

    if target_pixdims is not None:
        new_pix = np.array(target_pixdims)
    else:
        new_pix = np.array(img.pixdim)

    newimg, xform = resampleToPixdims(img, new_pix, order=order)

    return Image(newimg, xform=xform, header=img.header)


def order_voxels(voxels, direction='coronal'):
    """Re-order voxels according to slicing directions
    This is useful if we want to start caching slides
    Then we should be traversing voxels in roughly the same order
    as the slides

    :param voxels: Nx3 array
    :param direction: one of 'coronal', 'axial', or 'sagittal'
    :return: (N,3) reordered voxels as array
    """
    allowed_directions = ['coronal', 'axial', 'sagittal']
    assert direction.lower() in allowed_directions, f"allowed directions : {allowed_directions} but {direction} provided"

    if direction.lower() == 'coronal':
        idx = np.argsort(voxels[:,1])
    elif direction.lower() == 'axial':
        idx = np.argsort(voxels[:,2])
    else:
        idx = np.argsort(voxels[:,0])
    return voxels[idx,:]


# ---- Memory management ---- #
from functools import lru_cache
@lru_cache(maxsize=10)
def get_data(slide):
    """ Get image data with restricted caching
    :param slide: Image object
    :return:array
    """
    return Image(slide.dataSource).data

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
    theta = np.array(theta)/2.
    theta = np.pi - theta + angle*np.pi/180.
    theta = np.where(theta <= 0, theta + np.pi, theta)
    theta = np.where(theta > np.pi, theta - np.pi, theta)

    # theta = np.array(theta)/2.
    # theta = theta - angle*np.pi/180.
    # theta = np.where(theta <= 0, theta + np.pi, theta)
    # theta = np.where(theta > np.pi, theta - np.pi, theta)

    return theta
