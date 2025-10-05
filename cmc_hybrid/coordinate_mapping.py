#!/usr/bin/env python

# coordinate_mapping.py - function for mapping between dMRI and PSOCT coords
#
# Author: Silei Zhu, Saad Jbabdi, Amy Howard
#
# Copyright (C) 2025 University of Oxford
# SHBASECOPYRIGHT


import numpy as np
from fsl.transform.affine import concat, transform, invert
from scipy.ndimage import map_coordinates
from fsl.data.image import Image

from cmc_hybrid import utils

def slide_vox_intersect(voxel, slide, volume):
    """Test if voxel intersects with slide(s)

    WARNING!! This assumes that the data is Coronal!!!

    voxel      : list or array (single voxel)
    slide      : either Image object or list of Image objects
    volume_img : Image object

    Returns a bool or list of booleans
    """
    # If list provided, recursively run on list elements
    if type(slide) == list:
        return [slide_vox_intersect(voxel, s, volume) for s in slide]
    # get transformed from header
    world2pix = slide.getAffine('world', 'voxel')
    vox2world = volume.getAffine('voxel', 'world')
    vox2pix = concat(world2pix, vox2world)
    # test if near slide using y-coord
    # calc distance between slide and centre of the voxel
    # test that it is smaller than sqrt(3)/2*edge

    dist = np.abs(transform(voxel, vox2pix))[1] * slide.pixdim[1]
    return dist <= np.sqrt(3)/2. * np.max(volume.pixdim)

# Helper functions for cube operations/visualisation
def cube_vertices(XYZ, SL):
    """Gives the vertices of a cube as a list

    XYZ : coordinates of the centre
    SL  : length of the edges
    """
    Xc, Yc, Zc = XYZ
    return [[Xc + SL/2, Yc + SL/2, Zc + SL/2],
            [Xc + SL/2, Yc + SL/2, Zc - SL/2],
            [Xc + SL/2, Yc - SL/2, Zc + SL/2],
            [Xc + SL/2, Yc - SL/2, Zc - SL/2],
            [Xc - SL/2, Yc + SL/2, Zc + SL/2],
            [Xc - SL/2, Yc + SL/2, Zc - SL/2],
            [Xc - SL/2, Yc - SL/2, Zc + SL/2],
            [Xc - SL/2, Yc - SL/2, Zc - SL/2]]

def in_cube(coords, size, pos):
    """Test if a set of coordinates are inside a cube

    coords : Nx3 array
    size   : length of the cube's edges
    pos    : position of the centre of the cube

    returns booleans
    """
    pos = np.array(pos)
    size = np.array(size)
    return ((coords >= pos-size/2) & (coords < pos+size/2)).all(1)

def get_pixgrid(voxel, slide, volume):
    """Get a small grid of pixels surrounding a voxel
    This is a key function to avoid looking at all the coordinates
    of all the high resolution slide

    voxel : list or array (3 coordinates in voxel space)
    slide : Image object
    volume : Image object

    returns an array (Nx3)
    """
    # get transforms
    world2pix = slide.getAffine('world', 'voxel')
    vox2world = volume.getAffine('voxel', 'world')
    vox2pix   = concat(world2pix, vox2world)
    pix2vox   = invert(vox2pix)

    # get voxel cube
    voxel_bounds     = cube_vertices(voxel, 1)
    voxel_bounds_pix = transform(voxel_bounds, vox2pix)

    imin, jmin = np.min(voxel_bounds_pix, axis=0)[::2]
    imax, jmax = np.max(voxel_bounds_pix, axis=0)[::2]

    pixgrid = np.array( np.meshgrid( np.arange(imin, imax+1), np.arange(jmin, jmax+1)) )
    pixgrid = np.reshape(pixgrid, (2, -1)).T
    pixgrid = np.stack( (pixgrid[:,0], 0*pixgrid[:,0], pixgrid[:,1]), axis=1 )

    pixgrid2vox = transform(pixgrid, pix2vox)

    return pixgrid2vox, pixgrid

def vox_to_pix(vox, slides, volume):
    """Finds slide pixels within a voxel

    :param vox: 1D array of shape (3,)
    :param slides: list of Image objects
    :param volume: Image object
    :return:
    pixgrid : list of pixel coords in slide space
    voxgrid : list of pixel coords in voxel space
    pixgrid_data : list of slide data within the pixels
    """
    #1) find the slides that intersect the voxel
    slide_masks = slide_vox_intersect(vox, slides, volume)
    #2) for each slide found, get the pixgrid (pixels surrounding the voxel)
    pixgrid_all = []
    voxgrid_all = []
    pixgrid_data_all = []

    for idx in range(len(slides)):
        if slide_masks[idx]:
            pixgrid2vox, pixgrid = get_pixgrid(vox, slides[idx], volume)
            # select subset within the voxel
            mask        = in_cube(pixgrid2vox, 1, vox)
            pixgrid_all.extend(pixgrid[mask,:])
            voxgrid_all.extend(pixgrid2vox[mask,:])
            # get data
            data = utils.get_data(slides[idx])
            pixgrid_data_all.extend(map_coordinates(data, pixgrid[mask,:].T, order=0))

    return pixgrid_all, voxgrid_all, pixgrid_data_all


def slide_to_volume(slide, volume):
    """Calculate the xform from slide to volume

    :param slide: Image object
    :param volume: Image object
    :return: 2D array
    """
    pix2world = slide.getAffine('voxel', 'world')
    world2vox = volume.getAffine('world', 'voxel')
    pix2vox   = concat(world2vox, pix2world)

    return pix2vox[:3, :3]

def angle_to_vector(theta, xform=None):
    """
    Convert angle to 3D vector and re-orient into different coord system
    WARNING!!! This function assumes that the 2D data is CORONAL

    :param theta : 1D array or list
    :param xform : 2D array (2D-->3D transformation)
    :return:
    2D array (Nx3)    
    """
    v = np.stack([np.cos(theta), np.zeros_like(theta), -np.sin(theta)], axis=0)
    # apply xform
    if xform is not None:
        v = xform @ v
    # normalise
    v /= np.linalg.norm(v, axis=0, keepdims=True)
    return v.T

