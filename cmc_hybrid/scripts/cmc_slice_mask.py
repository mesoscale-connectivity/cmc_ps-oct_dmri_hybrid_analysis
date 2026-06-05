#!/usr/bin/env python


# cmc_slice_mask - create a brain mask based on slices
#
# Author: Silei Zhu, Saad Jbabdi, Amy Howard
#
# Copyright (C) 2025 University of Oxford
# SHBASECOPYRIGHT

import argparse


def parse_cmdline_args():
    p = argparse.ArgumentParser(description="CMC create brain mask")

    # Compulsory arguments
    p.add_argument("--slides", required=True, nargs="+",
                   help="slide files")
    p.add_argument("-o", "--out", required=True,
                   help="output mask")
    p.add_argument("--mask", required=True, help="brain mask")
    # Optional arguments
    p.add_argument("--roi", required=False, nargs=6, type=int,
                   help=("-roi <xmin> <xsize> <ymin> <ysize> <zmin> <zsize> : "
                         "only look inside roi (using voxel coordinates). "
                         "Inputting -1 for a size will set it to the full image extent for that dimension."))
    return p.parse_args()


def main():

    # Parse command-line arguments
    args = parse_cmdline_args()

    # ------------------------------------------------------------------------
    # More imports here so that running --help doesn't do unnecessary imports
    from fsl.data.image import Image
    import numpy as np
    from cmc_hybrid import utils
    # ------------------------------------------------------------------------

    # Load brain mask
    brain_img = Image(args.mask)
    brainmask = brain_img.data

    # Intersect with ROI
    brainmask = utils.prepare_mask(brainmask, args.roi)

    # Slides
    from fsl.utils.image.resample import resampleToReference
    slide_mask = 0 * brainmask
    for sl in args.slides:
        sl_vol = resampleToReference(Image(sl), brain_img)[0]
        slide_mask += np.array(sl_vol, dtype=brainmask.dtype)
        Image(slide_mask, header=brain_img.header).save('grotgrot')

    brainmask *= slide_mask

    # Save
    Image(brainmask, header=brain_img.header).save(args.out)


if __name__ == '__main__':
    main()
