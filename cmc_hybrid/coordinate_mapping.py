#!/usr/bin/env python

# coordinate_mapping.py - function for mapping between dMRI and PSOCT coords
#
# Author: Silei Zhu, Saad Jbabdi, Amy Howard, Vasilis Karlaftis
#
# Copyright (C) 2025 University of Oxford
# SHBASECOPYRIGHT


import numpy as np
from pathlib import Path
from fsl.transform.affine import concat, transform, invert
from fsl.transform.flirt import readFlirt, fromFlirt
from fsl.transform.nonlinear import DeformationField
from scipy.ndimage import map_coordinates
from fsl.data.image import Image
import fsl.data.constants as constants
from cmc_hybrid import utils
from pathlib import Path
import json

# TODO review and potentially replace this function with fslpy equivalent
def _matOrNifti(input):
    """Helper function to check either a .mat transformation matrix
      or a NIfTI warpfield

    :return:
    string equal to 'mat' or 'nii'
    boolean for file needs to be loaded (=1) or not (=0)
    """
    
    if input is None:
        return 'mat', 0

    if isinstance(input, Path):
        input = str(input)

    if isinstance(input, str):
        if input.endswith('.mat'):
            return 'mat', 1
        elif input.endswith('.nii') or input.endswith('.nii.gz'):
            if Image(input).intent in (constants.FSL_FNIRT_DISPLACEMENT_FIELD, constants.FSL_TOPUP_FIELD):
                return 'nii', 1
            raise ValueError(f"Invalid NIfTI warpfield file: indent={Image(input).intent}")
        else:
            raise ValueError("Input must be either a .mat or a .nii file.")
    elif isinstance(input, np.ndarray):
        return 'mat', 0
    # TODO add support on nibabel Nifti1Image objects
    elif isinstance(input, DeformationField):
        return 'nii', 0
    else:
        raise ValueError("Input must be either a mat array or a NIfTI warpfield file.")


def _world2pix(slide, world2world=None):
    world2pix = slide.getAffine('world', 'voxel')  # this is the DTI/MRI world
    if world2world is not None:
        world2pix = concat(world2pix, invert(world2world)) # PSOCT world-->PSOCT pix @ DTI world --> PSOCT world
    return world2pix


# old method superseded by slide_deck_vox_intersect
def slide_vox_intersect(voxel, volume, slide, slide2vol=None):
    """Test if voxel intersects with slide(s)

    WARNING!! This assumes that the data is Coronal!!!

    voxel       : list or array (single voxel)
    volume      : Image object
    slide       : either Image object or list of Image objects
    slide2vol   : PSOCT to DTI FLIRT affine matrix or FNIRT DeformationField object

    Returns a bool or list of booleans
    """
    # If list provided, recursively run on list elements
    if type(slide) == list:
        return [slide_vox_intersect(voxel, volume, s, slide2vol) for s in slide]
    # find if transform is linear or non-linear
    if isinstance(slide2vol, (type(None), np.ndarray)):
        world2pix = _world2pix(slide, slide2vol)
        vox2world = volume.getAffine('voxel', 'world')
        vox2pix   = concat(world2pix, vox2world)
    elif isinstance(slide2vol, DeformationField):
        vox2pix   = slide.getAffine('world', 'voxel')
        voxel     = slide2vol.transform(voxel, 'voxel', 'world')
    else:
        raise ValueError("Argument 'slide2vol' must be either a numpy array or a DeformationField file.")

    # test if near slide using y-coord
    # calc distance between slide and centre of the voxel
    # test that it is smaller than sqrt(3)/2*edge

    dist = np.abs(transform(voxel, vox2pix))[1] * slide.pixdim[1]
    return dist <= np.sqrt(3)/2. * np.max(volume.pixdim)


def slide_deck_vox_intersect(voxel, volume, slide_deck, slide2vol=None,direction="coronal"):
    """Test if voxel intersects with slide(s)

    WARNING!! This assumes that the data is Coronal!!!

    voxel       : list or array (single voxel)
    volume      : Image object
    slide_deck  : Image object or filename
    slide2vol   : PSOCT to DTI FLIRT affine matrix or FNIRT DeformationField object

    Returns a bool or list of booleans
    """
    # find if transform is linear or non-linear
    if isinstance(slide2vol, (type(None), np.ndarray)):
        world2pix = _world2pix(slide_deck, slide2vol)
        vox2world = volume.getAffine('voxel', 'world')
        vox2pix   = concat(world2pix, vox2world)
    elif isinstance(slide2vol, DeformationField):
        vox2pix   = slide_deck.getAffine('world', 'voxel')
        voxel     = slide2vol.transform(voxel, 'voxel', 'world')
    else:
        raise ValueError("Argument 'slide2vol' must be either a numpy array or a DeformationField file.")
    
    # test if near slide using y-coord
    # calc distance between slide and centre of the voxel
    # test that it is smaller than sqrt(3)/2*edge

    # TODO WIP - new solution is slightly different than current version
    # min_voxel = voxel.copy()
    # min_voxel[1] -= np.sqrt(3)/2
    # max_voxel = voxel.copy()
    # max_voxel[1] += np.sqrt(3)/2

    if direction == "coronal":
        ax, pd = 1, slide_deck.pixdim[1]
    elif direction == "sagittal":
        ax, pd = 0, slide_deck.pixdim[0]
    else:
        raise ValueError(f"direction must be 'coronal' or 'sagittal', got {direction!r}")
    dist = np.abs(transform(voxel, vox2pix))[ax] * pd

    dist = [dist - np.sqrt(3)/2. * np.max(volume.pixdim), dist + np.sqrt(3)/2. * np.max(volume.pixdim)]
    # convert distance back to slide indices
    # TODO add orientation information in if case here
    slides = [i / pd for i in dist]
    # slides = [np.abs(transform(min_voxel, vox2pix))[1], np.abs(transform(max_voxel, vox2pix))[1]]
    slides = np.arange(np.ceil(min(slides)), np.floor(max(slides)) + 1, dtype=int)
    return slides


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


def get_pixgrid(voxel, volume, slide, slide2vol=None, vol2slide=None, direction="coronal"):
    """Get a small grid of pixels surrounding a voxel
    This is a key function to avoid looking at all the coordinates
    of all the high resolution slide

    voxel       : list or array (3 coordinates in voxel space)
    volume      : Image object
    slide       : Image object
    slide2vol   : PSOCT to DTI FLIRT affine matrix or FNIRT DeformationField object
    vol2slide   : DTI to PSOCT FNIRT DeformationField object

    returns an array (Nx3)
    """
    # find if transform is linear or non-linear
    if isinstance(slide2vol, (type(None), np.ndarray)):
        world2pix = _world2pix(slide, slide2vol)
        vox2world = volume.getAffine('voxel', 'world')
        vox2pix   = concat(world2pix, vox2world)
    elif isinstance(slide2vol, DeformationField):
        vox2pix = slide.getAffine('world', 'voxel')
    else:
        raise ValueError("Argument 'slide2vol' must be either a numpy array or a DeformationField file.")

    # get voxel cube
    voxel_bounds     = cube_vertices(voxel, 1)
    if isinstance(slide2vol, DeformationField):
        voxel_bounds = slide2vol.transform(voxel_bounds, 'voxel', 'world')
    voxel_bounds_pix = transform(voxel_bounds, vox2pix)

    if direction == "coronal":
        imin, jmin = np.min(voxel_bounds_pix, axis=0)[::2]
        imax, jmax = np.max(voxel_bounds_pix, axis=0)[::2]
    elif direction == "sagittal":
        imin, jmin = np.min(voxel_bounds_pix, axis=0)[1:3]
        imax, jmax = np.max(voxel_bounds_pix, axis=0)[1:3]
    else:
        raise ValueError(f"direction must be 'coronal' or 'sagittal', got {direction!r}")

    pixgrid = np.array( np.meshgrid( np.arange(imin, imax+1), np.arange(jmin, jmax+1)) )
    pixgrid = np.reshape(pixgrid, (2, -1)).T

    if direction == "coronal":
        pixgrid = np.stack((pixgrid[:, 0], 0 * pixgrid[:, 0], pixgrid[:, 1]), axis=1)
    elif direction == "sagittal":
        pixgrid = np.stack((0 * pixgrid[:, 0], pixgrid[:, 0], pixgrid[:, 1]), axis=1)

    pixgrid2vox = transform(pixgrid, invert(vox2pix))
    if isinstance(vol2slide, DeformationField):
        pixgrid2vox = vol2slide.transform(pixgrid2vox, 'world', 'voxel')

    return pixgrid2vox, pixgrid


def vox_to_pix(vox, volume, slides, slide_deck=None, slide2vol=None, vol2slide=None):
    """Finds slide pixels within a voxel

    vox             : 1D array of shape (3,)
    volume          : Image object or filename
    slides          : list of Image objects
    slide_deck      : Image object or filename
    slide2vol       : PSOCT to DTI FLIRT affine matrix or FNIRT DeformationField object
    vol2slide       : DTI to PSOCT FNIRT DeformationField object

    :return:
    pixgrid         : list of pixel coords in slide space
    voxgrid         : list of pixel coords in voxel space
    pixgrid_data    : list of slide data within the pixels
    """

    volume = Image(volume)
    if slide_deck is not None:
        slide_deck = Image(slide_deck)
    
    #1) find the slides that intersect the voxel
    slide_masks = slide_vox_intersect(vox, volume, slides, slide2vol)
    #2) for each slide found, get the pixgrid (pixels surrounding the voxel)
    pixgrid_all = []
    voxgrid_all = []
    pixgrid_data_all = []

    for idx in range(len(slides)):
        if slide_masks[idx]:
            pixgrid2vox, pixgrid = get_pixgrid(vox, volume, slides[idx], slide2vol, vol2slide)

            # select subset within the voxel
            mask        = in_cube(pixgrid2vox, 1, vox)
            pixgrid_all.extend(pixgrid[mask,:])
            voxgrid_all.extend(pixgrid2vox[mask,:])
            # get data
            data = utils.get_data(slides[idx])
            pixgrid_data_all.extend(map_coordinates(data, pixgrid[mask,:].T, order=0))

    return pixgrid_all, voxgrid_all, pixgrid_data_all


def vox_to_pix_slidedeck(vox, volume, ori_slides_dir, ret_slides_dir, slide_deck, slide_mapping, slide2vol=None, vol2slide=None, direction="coronal"):
    """Finds slide pixels within a voxel

    vox             : 1D array of shape (3,)
    volume          : Image object or filename
    ori_slides_dir  : Directory of PSOCT orientation slides
    ret_slides_dir  : Directory of PSOCT retardance slides
    slide_deck      : Image object or filename
    slide_mapping   : a dict of slidedeck to slides mapping
    slide2vol       : PSOCT to DTI FLIRT affine matrix or FNIRT DeformationField object
    vol2slide       : DTI to PSOCT FNIRT DeformationField object

    :return:
    pixgrid         : list of pixel coords in slide space
    voxgrid         : list of pixel coords in voxel space
    slide_index     : list of slide indices within the voxel
    pixgrid_data    : list of orientation data within the pixels
    retardance_data : list of retardance data within the pixels
    """

    volume = Image(volume)
    slide_deck = Image(slide_deck)

    #1) find the slides that intersect the voxel
    slide_masks = slide_deck_vox_intersect(vox, volume, slide_deck, slide2vol, direction=direction)
    #2) for each slide found, get the pixgrid (pixels surrounding the voxel)
    pixgrid_all = []
    voxgrid_all = []
    pixgrid_data_all = []
    slide_index = []

    if ret_slides_dir is None:
        retardance_data = None
    else:
        retardance_data = []

    for idx in slide_masks:
        if str(idx) not in slide_mapping.keys():
            continue
        slide = next(Path(ori_slides_dir).glob(slide_mapping[str(idx)]), None)
        if slide is not None:
            slide = Image(slide)
        else:
            continue
        pixgrid2vox, pixgrid = get_pixgrid(vox, volume, slide, slide2vol, vol2slide, direction=direction)

        # select subset within the voxel
        mask        = in_cube(pixgrid2vox, 1, vox)
        pixgrid_all.extend(pixgrid[mask,:])
        voxgrid_all.extend(pixgrid2vox[mask,:])
        # get data
        data = utils.get_data(slide)
        pixgrid_data_all.append(map_coordinates(data, pixgrid[mask,:].T, order=0))
        # keep track of slide index
        slide_index.append(idx)

        if ret_slides_dir is not None:
            slide = next(Path(ret_slides_dir).glob(slide_mapping[str(idx)]), None)
            if slide is not None:
                retardance = utils.get_data(Image(slide))
                retardance_values = map_coordinates(retardance, pixgrid[mask,:].T, order=0)
                retardance_data.extend(retardance_values)

    return pixgrid_all, voxgrid_all, slide_index, pixgrid_data_all, retardance_data


# # TODO remove this and replace with the new functions
# def slide_to_volume(volume, slide, slide_deck=None, slide2vol=None):
#     """Calculate the xform from slide to volume

#     volume      : Image object
#     slide       : Image object
#     slide_deck  : Image object or filename
#     slide2vol   : filename of PSOCT_to_DTI.mat

#     :return: 2D array
#     """
#     # TODO review if this needs changing
#     pix2world = slide.getAffine('voxel', 'world')
#     if slide2vol is not None and slide_deck is not None:
#         input, load_flag = _matOrNifti(slide2vol)
#         if input == 'mat':
#             pix2world = concat(slide2vol, pix2world)
#         elif input == 'nii':
#             raise ValueError(f"Incompatible input type '{input}' for calculating the xform.")
#         else:
#             raise ValueError("Argument 'slide2vol' must be either a .mat or a .nii file.")

#     world2vox = volume.getAffine('world', 'voxel')
#     pix2vox   = concat(world2vox, pix2world)

#     return pix2vox[:3, :3]

def slidedeck_to_volume(vecs, vox, warp=None):
    """
    Align the vectors from slidedeck to dMRI space
    """
    if warp is None:
        return vecs
    
    jx, jy, jz = [np.gradient(warp.data[...,i], *warp.pixdim[:3]) for i in range(3)]

    # Nonlinear Jacobian at this voxel
    F  = affine_from_jac([int(np.round(x)) for x in vox], jx, jy, jz, warp.isNeurological())
    
    # slidedeck -> dMRI re-orientation
    all_v = concat(invert(F), vecs)
    all_v /= np.linalg.norm(all_v, axis=0, keepdims=True)

    return all_v.T

def slide_to_deck(theta, slide_index, ori_slides_dir, slide_deck, slide_mapping, direction="coronal"):
    """ 
    Align the vectors from slide to slidedeck space
    """

    all_v = []

    for sl, angles in zip(slide_index, theta):
        angles = utils.fudge_psoct_orientation(angles)

        slide = next(Path(ori_slides_dir).glob(slide_mapping[str(sl)]), None)
        if slide is not None:
            slide = Image(slide)
        else:
            continue
        
        # slide deck -> slide re-orientation
        F = concat(slide.getAffine('world', 'voxel'), slide_deck.getAffine('voxel', 'world'))
        v = angle_to_vector(angles, invert(F[:3,:3]), direction=direction)
        all_v.append(v)
    all_v = np.concatenate(all_v,axis=1)

    return all_v

def affine_from_jac(vox, jx, jy, jz, is_neuro=False):
    x, y, z = vox
    x_sign = 1.
    if is_neuro:
        x_sign = -1.
    Jw  = [[x_sign*jx[0][x,y,z], jx[1][x,y,z], jx[2][x,y,z]],
           [x_sign*jy[0][x,y,z], jy[1][x,y,z], jy[2][x,y,z]], 
           [x_sign*jz[0][x,y,z], jz[1][x,y,z], jz[2][x,y,z]]]
    Jw  = np.array(Jw)
    aff = np.eye(3) + Jw    
    return aff

def angle_to_vector(theta, xform=None, direction="coronal"):
    """
    Convert angle to 3D vector and re-orient into different coord system
    WARNING!!! This function assumes that the 2D data is CORONAL

    :param theta : 1D array or list
    :param xform : 2D array (2D-->3D transformation)
    :return:
    2D array (Nx3)    
    """
    if direction == "coronal":
        v = np.stack([np.cos(theta), np.zeros_like(theta), np.sin(theta)], axis=0)
    elif direction == "sagittal":
        v = np.stack([np.zeros_like(theta), np.cos(theta), -np.sin(theta)], axis=0)

    # apply xform
    if xform is not None:
        v = xform @ v
    # normalise
    v /= np.linalg.norm(v, axis=0, keepdims=True)
    return v

