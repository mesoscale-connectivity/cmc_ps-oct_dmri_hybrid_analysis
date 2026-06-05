#!/usr/bin/env python


# cmc_hybrid - runs hybrid modelling of dMRI and tractography
#
# Author: Silei Zhu, Saad Jbabdi, Amy Howard, Vasilis Karlaftis
#
# Copyright (C) 2025 University of Oxford
# SHBASECOPYRIGHT

import argparse


def parse_cmdline_args():
    p = argparse.ArgumentParser(description="CMC hybrid per-voxel processor")

    # Compulsory arguments
    p.add_argument("--ori_slides_dir", required=True,
                   help="Directory of PSOCT Orientation slides")
    p.add_argument("--slidedeck", required=True,
                   help="PSOCT slidedeck file")
    p.add_argument("--slide_mapping", required=True,
                   help="JSON file of slidedeck to slides mapping")
    p.add_argument("--bpx", required=True,
                   help="location of the bedpostX folder")
    p.add_argument("-o", "--out", required=True,
                   help="prefix for output files")
    # Optional arguments
    p.add_argument("--ret_slides_dir", required=False,
                   help="Directory of PSOCT Retardance slides")
    p.add_argument("--retardance_threshold", required=False, type=float, default=19,
                   help="Threshold for Rretardance images")
    p.add_argument("--slide2vol", required=False,
                   help="psoct-to-mri transformation matrix or warp field")
    p.add_argument("--vol2slide", required=False,
                   help="mri-to-psoct transformation matrix or warp field")
    p.add_argument("--roi", required=False, nargs=6, type=int,
                   help=("-roi <xmin> <xsize> <ymin> <ysize> <zmin> <zsize> : "
                         "only look inside roi (using voxel coordinates). "
                         "Inputting -1 for a size will set it to the full image extent for that dimension."))
    p.add_argument("--mask", required=False, help="brain mask")
    p.add_argument("-j", "--jobs", type=int,
                   help="parallel workers (default=max available)")
    p.add_argument("--output_type", required=False, type=str, default='h-fod',
                   help="type of output: h-fod (default) or h-dyad or psoct-dyad or psoct-fod, inplane-dMRI")
    p.add_argument("--deterministic", required=False, action='store_true',
                   help="deterministic out-of-plane sampling (only relevant for FOD output)")
    p.add_argument("--f_thr", required=False, type=float,
                   help="f-threshold")
    p.add_argument("-s", "--resolution", nargs=3, type=float, metavar=('SX', 'SY', 'SZ'),
                   help="target resolution, e.g. -s 0.4 0.4 0.4", default=[0.4, 0.4, 0.4])
    p.add_argument("--slice_axis", required=False, type=str, default='coronal',
                   help="slice direction. one of 'coronal' (default), 'sagittal', or 'axial'")
    p.add_argument("--SHorder", required=False, type=int, default=8,
                   help="spherical harmonics order. default=8.")
    p.add_argument("--verbose", required=False, action='store_true',
                   help='print out messages')
    return p.parse_args()


def main():
    # Parse command-line arguments
    args = parse_cmdline_args()
    check_args(args)

    if args.verbose:
        print("Begin.")
    # ------------------------------------------------------------------------
    # More imports here so that running --help doesn't do unnecessary imports
    from fsl.data.image import Image
    import os
    import json
    from glob import glob
    import numpy as np
    from joblib import Parallel, delayed, parallel_backend
    from functools import partial
    from cmc_hybrid import utils, fod
    from cmc_hybrid import coordinate_mapping as cm
    # ------------------------------------------------------------------------

    # Find slides
    if args.verbose:
        print("...Read slides")

    slide_deck = Image(args.slidedeck)

    with open(args.slide_mapping, 'r') as file:
        slide_map = json.load(file)

    # Load image files from database
    if args.verbose:
        print("...Prepare brain mask...")
    # Prepare mask
    maskfile = args.mask
    if args.mask is None:
        maskfile = os.path.join(args.bpx, 'nodif_brain_mask')

    mask_img = utils.prepare_mask_slidedeck(maskfile, args.roi, args.resolution, slide_deck,
                                            slide_direction=args.slice_axis, matOrWarp=args.slide2vol)
    mask     = mask_img.data

    if args.verbose:
        print(f"...new mask dimensions: shape={mask_img.shape}, pixdim={mask_img.pixdim}")

    # Prepare voxels based on roi and mask
    # voxels contains all the voxel coordinates within the mask and the roi
    voxels = np.argwhere(mask)
    # reorder by slicing axis for more efficient caching
    voxels = utils.order_voxels(voxels, args.slice_axis)

    if len(voxels) == 0:
        raise Exception("Found no voxels to process. Please check the mask and roi definitions.")
    if args.verbose:
        print(f"...Processing {len(voxels)} voxels")

    # voxels in dMRI space:
    from fsl.transform.affine import concat, transform
    mask2diff = concat(
        Image(os.path.join(args.bpx, 'nodif_brain_mask')).getAffine('world', 'voxel'),
        mask_img.getAffine('voxel', 'world')
    )
    voxels_diff = transform(voxels, mask2diff)
    voxels_both = zip(voxels, voxels_diff)  # this has both sets of voxels

    # Global variables
    # All the big 4D files go here
    if args.verbose:
        print("...Loading bpx samples")

    # no need to load bpx samples if we are only interested in the psoct orientations
    if args.output_type == 'psoct-dyad':
        if args.verbose:
            print('......not loaded (only psoct orientations are analysed)')

        ths, phs, fs = [], [], []
    else:
        ths = [Image(x).data for x in sorted(glob(os.path.join(args.bpx, 'merged_th?samples.nii.gz')))]
        phs = [Image(x).data for x in sorted(glob(os.path.join(args.bpx, 'merged_ph?samples.nii.gz')))]
        fs  = [Image(x).data for x in sorted(glob(os.path.join(args.bpx, 'merged_f?samples.nii.gz')))]
        if args.verbose:
            print('......loaded')

    # --------- Parallelised functions --------------- #
    # def process_voxel(vox_coord, ths, phs, fs):
    #     x,y,z = vox_coord
    #     th_samples = [th[x,y,z,:] for th in ths]
    #     ph_samples = [ph[x,y,z,:] for ph in phs]
    #     f_samples  = [f[x,y,z,:] for f in fs]
    #     vecs       = [utils.pol2cart(th,ph) for th,ph in zip(th_samples, ph_samples)]
    #     dyads      = [utils.make_dyads(v) for v in vecs]
    #     return dyads

    def process_voxel_hybrid(vox_both, ths, phs, fs,
                             volume, ori_slides_dir, ret_slides_dir, retardance_threshold,
                             slide_deck, slide_mapping, slice_axis, slide2vol, vol2slide,
                             output_type, deterministic, f_thr, SHorder):
        """
        For now, simply find the slides that intersect, the pixels in the voxel, and count them
        return the number of pixels in the voxel
        """
        # Get pixels and data
        vox_coord, vox_diff = vox_both
        pixgrid, voxgrid, slide_index, theta, retardance = cm.vox_to_pix_slidedeck(vox_diff, volume, ori_slides_dir,
                                                                                   ret_slides_dir, slide_deck,
                                                                                   slide_mapping, slide2vol, vol2slide,
                                                                                   direction=slice_axis)
        # theta here ^ is defined in the pixel / slide space

        # flatten theta for the next check
        flattened_theta = [item for sublist in theta
                           for item in (sublist if isinstance(sublist, np.ndarray) else [sublist])]
        # need at least three values to get a 3x3 SVD later
        if len(flattened_theta) < 3:
            if output_type in ['h-dyad', 'psoct-dyad', 'inplane-dMRI']:
                return np.zeros(3)
            else:
                n = fod.SHcoeffLen(SHorder)
                return np.zeros(n)

        # convert vecs from slide to slidedeck space
        vecs = cm.slide_to_deck(theta, slide_index, ori_slides_dir, slide_deck, slide_mapping, direction=slice_axis)
        # convert vecs from slidedeck to dMRI space
        vecs = cm.slidedeck_to_volume(vecs, vox_diff, slide2vol)

        # stop here if only psoct-dyad are requested
        if output_type == 'psoct-dyad':
            dyad  = utils.make_dyads(vecs)
            return dyad
        if output_type == 'psoct-fod':
            fod_coef  = fod.fit_sh_fod(vecs, max_order=SHorder, symmetric=True, weights=None,
                                       kde_bw=20., normalise=False, output_kde=False)
            return np.squeeze(np.array([fod_coef]))

        # calc_hybrid orientations
        x, y, z    = np.array(vox_diff).astype(int)
        th_samples = np.array([th[x, y, z, :] for th in ths]).flatten()
        ph_samples = np.array([ph[x, y, z, :] for ph in phs]).flatten()
        f_samples  = np.array([f[x, y, z, :]  for f in fs]).flatten()
        if f_thr is not None:
            f_samples[f_samples < f_thr] = 0.

        h_vecs = fod.hybrid_vecs(th_samples, ph_samples, f_samples, vecs, retardance, retardance_threshold,
                                 output_type, weighted=True, deterministic=deterministic)

        # calc dyads
        if output_type == 'h-dyad' or output_type == 'inplane-dMRI':
            dyads  = utils.make_dyads(h_vecs)
            return dyads
        elif output_type == 'h-fod':
            fod_coef  = fod.fit_sh_fod(h_vecs, max_order=SHorder, symmetric=True, normalise=True)
            return fod_coef
        else:
            raise Exception("output_type must be one of 'h-fod', 'h-dyad', 'inplane-dMRI' 'psoct-fod' or 'psoct-dyad'")

    # ------------------------------------------------------------------------------

    # Run processing for each voxel (parallelise)
    if args.jobs:
        n_workers = args.jobs
    else:
        import multiprocessing as mp
        n_workers = mp.cpu_count() - 1

    # Load transformation matrices and warp fields
    # Check the slide2vol argument type
    if args.slide2vol is not None:
        format, _ = cm._matOrNifti(args.slide2vol)
        if format == 'mat':
            from fsl.transform.flirt import readFlirt, fromFlirt
            slide2vol = readFlirt(args.slide2vol)
            slide2vol = fromFlirt(slide2vol, slide_deck, Image(maskfile), from_='world', to='world')
        elif format == 'nii':
            from fsl.transform.fnirt import readFnirt
            slide2vol = readFnirt(args.slide2vol, slide_deck, Image(maskfile))
            # load Image data to expedite the parallel processing
            slide2vol.transform([0, 0, 0], 'voxel', 'world')
        else:
            print("Argument 'slide2vol' must be either a .mat or a .nii file. Ignoring this argument.")
            slide2vol = None
    else:
        slide2vol = None
    # Check the vol2slide argument type
    if args.vol2slide is not None:
        format, _ = cm._matOrNifti(args.vol2slide)
        if format == 'nii':
            from fsl.transform.fnirt import readFnirt
            vol2slide = readFnirt(args.vol2slide, Image(maskfile), slide_deck)
            # load Image data to expedite the parallel processing
            vol2slide.transform([0, 0, 0], 'world', 'voxel')
        else:
            print("Argument 'vol2slide' must be either a .nii file. Ignoring this argument.")
            vol2slide = None
    else:
        vol2slide = None

    func = partial(process_voxel_hybrid, ths=ths, phs=phs, fs=fs, volume=Image(maskfile),
                   ori_slides_dir=args.ori_slides_dir, ret_slides_dir=args.ret_slides_dir,
                   retardance_threshold=args.retardance_threshold,
                   slide_deck=slide_deck, slide_mapping=slide_map, slice_axis=args.slice_axis,
                   slide2vol=slide2vol, vol2slide=vol2slide, output_type=args.output_type,
                   deterministic=args.deterministic, f_thr=args.f_thr, SHorder=args.SHorder)

    if args.verbose:
        print(
            f"""
            Running process_voxel_hybrid with the following arguments:
            output_type   : {args.output_type},
            deterministic : {args.deterministic},
            f_thr         : {args.f_thr},
            SHorder       : {args.SHorder}

            volume        : {mask_img.dataSource},
            ori_slides_dir    : {args.ori_slides_dir},
            """
        )

    results = []
    with parallel_backend("threading", n_jobs=n_workers):
        verbosity = 13 if args.verbose else 0
        results = Parallel(verbose=verbosity)(delayed(func)(vox) for vox in voxels_both)

    # Run Sequentially (for testing)
    # for vox in voxels_both:
    #    print(vox)
    #    results.append( func(vox) )

    # Save output
    if args.verbose:
        print('...saving')
    results = np.array(results)
    save_results(results, voxels, mask_img, args.out)

    # Finish.
    if args.verbose:
        print("Done.")


# -------------- Other functions ------------------------- #
def save_results(results, voxels, mask_img, outfile):
    """Save results are a NIFTI file

    :param results: array
    :param voxels: array
    :param mask_img: Image obj
    :param outfile: str
    :return:
    """
    import numpy as np
    from fsl.data.image import Image
    n         = results.shape[-1]
    out_shape = mask_img.shape+(n,)

    out_array = np.zeros(out_shape)
    out_array[voxels[:, 0], voxels[:, 1], voxels[:, 2], :] = results

    Image(out_array, header=mask_img.header).save(outfile)

    return


def check_args(args):
    """Check cmd line args
    """
    allowed_output_types = ['h-fod', 'h-dyad', 'psoct-dyad', 'inplane-dMRI', 'psoct-fod']
    assert args.output_type in allowed_output_types, \
        f"output_type must be one of {allowed_output_types}, but value given is {args.output_type}"

    allowed_slice_axes = ['coronal', 'sagittal']
    assert args.slice_axis.lower() in allowed_slice_axes, \
        f"slice_axis must be one of {allowed_slice_axes}, but value given is {args.slice_axis}"


if __name__ == '__main__':
    main()
