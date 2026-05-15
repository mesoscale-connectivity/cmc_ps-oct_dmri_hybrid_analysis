from cmc_hybrid import coordinate_mapping as cm
from pathlib import Path
import numpy as np
import json
from fsl.data.image import Image
from fsl.transform.flirt import readFlirt, fromFlirt
from fsl.transform.fnirt import readFnirt

testsPath = Path(__file__).parent

def test_vox_to_pix():
    volume = Image(testsPath / 'testdata/volume')
    slices = [Image(x) for x in [testsPath / 'testdata/slice1', testsPath / 'testdata/slice2']]
    vox    = [25, 28, 9]
    a,b,c  = cm.vox_to_pix(vox, volume, slices)
    assert len(a)>0

def test_vox_to_pix_in_native():
    vox    = [24, 22, 9]
    volume = Image(testsPath / 'testdata/volume_in_DTI')
    slices = [Image(x) for x in [testsPath / 'testdata/slice_085_in_DTI', testsPath / 'testdata/slice_086_in_DTI']]
    a1,b1,c1  = cm.vox_to_pix(vox, volume, slices)
    slide_deck  = Image(testsPath / 'testdata/slidedeck_in_PSOCT')
    psoct2dti = testsPath / 'testdata/PSOCT_to_DTI.mat'
    psoct2dti = readFlirt(psoct2dti)
    psoct2dti = fromFlirt(psoct2dti, slide_deck, volume, from_='world', to='world')

    slices = [Image(x) for x in [testsPath / 'testdata/slice_085_in_PSOCT', testsPath / 'testdata/slice_086_in_PSOCT']]
    a2,b2,c2  = cm.vox_to_pix(vox, volume, slices, slide_deck, psoct2dti)
    a3,b3,c3  = cm.vox_to_pix(vox, volume, slices)

    assert len(a1)>0 and len(a2)>0
    assert len(a1) == len(a2)
    assert np.allclose(a1, a2)
    assert np.allclose(b1, b2)
    assert np.allclose(c1, c2)
    assert len(a2) != len(a3) or not np.allclose(a2, a3)
    assert len(b2) != len(b3) or not np.allclose(b2, b3)
    assert len(c2) != len(c3) or not np.allclose(c2, c3)

def test_vox_to_pix_warp():
    vox    = [24, 22, 9]
    volume = Image(testsPath / 'testdata/volume_in_DTI')
    slide_deck  = Image(testsPath / 'testdata/slidedeck_in_PSOCT')
    psoct2dti = testsPath / 'testdata/PSOCT_to_DTI.mat'
    psoct2dti = readFlirt(psoct2dti)
    psoct2dti = fromFlirt(psoct2dti, slide_deck, volume, from_='world', to='world')
    slices = [Image(x) for x in [testsPath / 'testdata/slice_085_in_PSOCT', testsPath / 'testdata/slice_086_in_PSOCT']]
    a1,b1,c1  = cm.vox_to_pix(vox, volume, slices, slide_deck, psoct2dti)
    psoct2dti = testsPath / 'testdata/PSOCT_to_DTI_warpfield.nii.gz'
    dti2psoct = testsPath / 'testdata/DTI_to_PSOCT_warpfield.nii.gz'
    psoct2dti = readFnirt(psoct2dti, slide_deck, volume)
    dti2psoct = readFnirt(dti2psoct, volume, slide_deck)
    a2,b2,c2  = cm.vox_to_pix(vox, volume, slices, slide_deck, psoct2dti, dti2psoct)

    assert len(a1)>0 and len(a2)>0
    assert np.allclose(np.array(a1).mean(axis=0), np.array(a2).mean(axis=0), atol=3)
    assert np.allclose(np.array(b1).mean(axis=0), np.array(b2).mean(axis=0), atol=1)
    assert np.allclose(np.array(c1).mean(axis=0), np.array(c2).mean(axis=0), atol=2)
    
def test_vox_to_pix_slidedeck_in_native(tmp_path):
    vox    = [24, 22, 9]
    volume = Image(testsPath / 'testdata/volume_in_DTI')
    slides_dir = testsPath / 'testdata'

    # set variables for DTI space
    with open(tmp_path / "slidedeck_slice_mapping.json", "w") as f:
        json.dump({i: f"slice_{str(145-i).zfill(3)}_in_DTI.nii.gz" for i in range(145)}, f, indent=2)
    slide_mapping = tmp_path / 'slidedeck_slice_mapping.json'
    with open(slide_mapping, 'r') as file:
        slide_mapping = json.load(file)
    slide_deck  = Image(testsPath / 'testdata/slidedeck_in_DTI')
    a1,b1,c1,d1,e1  = cm.vox_to_pix_slidedeck(vox, volume, slides_dir, slides_dir, slide_deck, slide_mapping,direction="coronal")
    flattened_d1 = [item for sublist in d1 for item in (sublist if isinstance(sublist, np.ndarray) else [sublist])]

    # set variables for PSOCT space
    slide_mapping = testsPath / 'testdata/slidedeck_slice_mapping.json'
    with open(slide_mapping, 'r') as file:
        slide_mapping = json.load(file)
    slide_deck  = Image(testsPath / 'testdata/slidedeck_in_PSOCT')
    psoct2dti = testsPath / 'testdata/PSOCT_to_DTI.mat'
    psoct2dti = readFlirt(psoct2dti)
    psoct2dti = fromFlirt(psoct2dti, slide_deck, volume, from_='world', to='world')

    a2,b2,c2,d2,e2  = cm.vox_to_pix_slidedeck(vox, volume, slides_dir, slides_dir, slide_deck, slide_mapping, psoct2dti,direction="coronal")
    flattened_d2 = [item for sublist in d2 for item in (sublist if isinstance(sublist, np.ndarray) else [sublist])]
    a3,b3,c3,d3,e3  = cm.vox_to_pix_slidedeck(vox, volume, slides_dir, slides_dir, slide_deck, slide_mapping,direction="coronal")
    flattened_d3 = [item for sublist in d3 for item in (sublist if isinstance(sublist, np.ndarray) else [sublist])]

    assert len(a1)>0 and len(a2)>0
    assert len(a1) == len(a2)
    assert np.allclose(a1, a2)
    assert np.allclose(b1, b2)
    assert c1 == c2
    assert np.allclose(flattened_d1, flattened_d2)
    assert np.allclose(e1, e2)
    assert len(a2) != len(a3) or not np.allclose(a2, a3)
    assert len(b2) != len(b3) or not np.allclose(b2, b3)
    assert len(c2) != len(c3) or not np.allclose(c2, c3)
    assert len(flattened_d2) != len(flattened_d3) or not np.allclose(flattened_d2, flattened_d3)
    assert len(e2) != len(e3) or not np.allclose(e2, e3)

def test_vox_to_pix_vs_vox_to_pix_slidedeck_in_native():
    vox    = [24, 22, 9]
    volume = Image(testsPath / 'testdata/volume_in_DTI')
    slides_dir = testsPath / 'testdata'
    # pass slices in reverse order to match how vox_to_pix_slidedeck reads them from slide_mapping
    slices = [Image(x) for x in [slides_dir / 'slice_086_in_PSOCT', slides_dir / 'slice_085_in_PSOCT']]
    slide_deck  = Image(testsPath / 'testdata/slidedeck_in_PSOCT')
    slide_mapping = testsPath / 'testdata/slidedeck_slice_mapping.json'
    with open(slide_mapping, 'r') as file:
        slide_mapping = json.load(file)
    psoct2dti = testsPath / 'testdata/PSOCT_to_DTI.mat'
    psoct2dti = readFlirt(psoct2dti)
    psoct2dti = fromFlirt(psoct2dti, slide_deck, volume, from_='world', to='world')

    a1,b1,c1  = cm.vox_to_pix(vox, volume, slices, slide_deck, psoct2dti)
    a2,b2,_,c2,_  = cm.vox_to_pix_slidedeck(vox, volume, slides_dir, slides_dir, slide_deck, slide_mapping, psoct2dti,direction="coronal")
    flattened_c2 = [item for sublist in c2 for item in (sublist if isinstance(sublist, np.ndarray) else [sublist])]

    assert len(a1)>0 and len(a2)>0
    assert len(a1) == len(a2)
    assert np.allclose(a1, a2)
    assert np.allclose(b1, b2)
    assert np.allclose(c1, flattened_c2)

def test_vox_to_pix_slidedeck_warp():
    vox    = [24, 22, 9]
    volume = Image(testsPath / 'testdata/volume_in_DTI')
    slides_dir = testsPath / 'testdata'
    slide_mapping = testsPath / 'testdata/slidedeck_slice_mapping.json'
    with open(slide_mapping, 'r') as file:
        slide_mapping = json.load(file)
    slide_deck  = Image(testsPath / 'testdata/slidedeck_in_PSOCT')

    # set variables for linear registration
    psoct2dti = testsPath / 'testdata/PSOCT_to_DTI.mat'
    psoct2dti = readFlirt(psoct2dti)
    psoct2dti = fromFlirt(psoct2dti, slide_deck, volume, from_='world', to='world')
    a1,b1,c1,d1,e1  = cm.vox_to_pix_slidedeck(vox, volume, slides_dir, slides_dir, slide_deck, slide_mapping, psoct2dti,direction="coronal")
    flattened_d1 = [item for sublist in d1 for item in (sublist if isinstance(sublist, np.ndarray) else [sublist])]

    # set variables for non-linear registration
    psoct2dti = testsPath / 'testdata/PSOCT_to_DTI_warpfield.nii.gz'
    dti2psoct = testsPath / 'testdata/DTI_to_PSOCT_warpfield.nii.gz'
    psoct2dti = readFnirt(psoct2dti, slide_deck, volume)
    dti2psoct = readFnirt(dti2psoct, volume, slide_deck)
    a2,b2,c2,d2,e2  = cm.vox_to_pix_slidedeck(vox, volume, slides_dir, slides_dir, slide_deck, slide_mapping, psoct2dti, dti2psoct,direction="coronal")
    flattened_d2 = [item for sublist in d2 for item in (sublist if isinstance(sublist, np.ndarray) else [sublist])]

    assert len(a1)>0 and len(a2)>0
    assert np.allclose(np.array(a1).mean(axis=0), np.array(a2).mean(axis=0), atol=3)
    assert np.allclose(np.array(b1).mean(axis=0), np.array(b2).mean(axis=0), atol=1)
    assert c1 == c2
    assert np.allclose(np.array(flattened_d1).mean(axis=0), np.array(flattened_d2).mean(axis=0), atol=2)
    assert np.allclose(np.array(e1).mean(axis=0), np.array(e2).mean(axis=0), atol=2)

def test_vox_to_pix_vs_vox_to_pix_slidedeck_warp():
    vox    = [24, 22, 9]
    volume = Image(testsPath / 'testdata/volume_in_DTI')
    slides_dir = testsPath / 'testdata'
    # pass slices in reverse order to match how vox_to_pix_slidedeck reads them from slide_mapping
    slices = [Image(x) for x in [slides_dir / 'slice_086_in_PSOCT', slides_dir / 'slice_085_in_PSOCT']]
    slide_deck  = Image(testsPath / 'testdata/slidedeck_in_PSOCT')
    slide_mapping = testsPath / 'testdata/slidedeck_slice_mapping.json'
    with open(slide_mapping, 'r') as file:
        slide_mapping = json.load(file)
    psoct2dti = testsPath / 'testdata/PSOCT_to_DTI_warpfield.nii.gz'
    dti2psoct = testsPath / 'testdata/DTI_to_PSOCT_warpfield.nii.gz'
    psoct2dti = readFnirt(psoct2dti, slide_deck, volume)
    dti2psoct = readFnirt(dti2psoct, volume, slide_deck)
    a1,b1,c1  = cm.vox_to_pix(vox, volume, slices, slide_deck, psoct2dti, dti2psoct)
    a2,b2,_,c2,_  = cm.vox_to_pix_slidedeck(vox, volume, slides_dir, slides_dir, slide_deck, slide_mapping, psoct2dti, dti2psoct,direction="coronal")
    flattened_c2 = [item for sublist in c2 for item in (sublist if isinstance(sublist, np.ndarray) else [sublist])]

    assert len(a1)>0 and len(a2)>0
    assert len(a1) == len(a2)
    assert np.allclose(a1, a2)
    assert np.allclose(b1, b2)
    assert np.allclose(c1, flattened_c2)

def test_slide_vox_intersect():
    slide  = Image(testsPath / 'testdata/slice2')
    volume = Image(testsPath / 'testdata/volume')
    a = cm.slide_vox_intersect([25, 28, 9], volume, slide)
    assert a

    a = cm.slide_vox_intersect([25, 10, 9], volume, slide)
    assert not a

def test_slide_vox_intersect_in_native():
    vox = [24, 22, 9]
    volume = Image(testsPath / 'testdata/volume_in_DTI')
    slide = Image(testsPath / 'testdata/slice_085_in_DTI')
    a1 = cm.slide_vox_intersect(vox, volume, slide)
    slide_deck  = Image(testsPath / 'testdata/slidedeck_in_PSOCT')
    psoct2dti = testsPath / 'testdata/PSOCT_to_DTI.mat'
    psoct2dti = readFlirt(psoct2dti)
    psoct2dti = fromFlirt(psoct2dti, slide_deck, volume, from_='world', to='world')
    slide = Image(testsPath / 'testdata/slice_085_in_PSOCT')
    a2 = cm.slide_vox_intersect(vox, volume, slide, psoct2dti)
    a3 = cm.slide_vox_intersect(vox, volume, slide)
    assert a1 and a2
    assert a1 == a2
    assert a2 != a3

    vox = [25, 28, 9]
    slide = Image(testsPath / 'testdata/slice_085_in_DTI')
    a1 = cm.slide_vox_intersect(vox, volume, slide)
    slide = Image(testsPath / 'testdata/slice_085_in_PSOCT')
    a2 = cm.slide_vox_intersect(vox, volume, slide, psoct2dti)
    assert not a1 and not a2
    assert a1 == a2

def test_slide_vox_intersect_warp():
    vox = [24, 22, 9]
    volume = Image(testsPath / 'testdata/volume_in_DTI')
    slide_deck  = Image(testsPath / 'testdata/slidedeck_in_PSOCT')
    psoct2dti = testsPath / 'testdata/PSOCT_to_DTI.mat'
    psoct2dti = readFlirt(psoct2dti)
    psoct2dti = fromFlirt(psoct2dti, slide_deck, volume, from_='world', to='world')
    slide = Image(testsPath / 'testdata/slice_085_in_PSOCT')
    a1 = cm.slide_vox_intersect(vox, volume, slide, psoct2dti)
    psoct2dti = testsPath / 'testdata/PSOCT_to_DTI_warpfield.nii.gz'
    psoct2dti = readFnirt(psoct2dti, slide_deck, volume)
    a2 = cm.slide_vox_intersect(vox, volume, slide, psoct2dti)

    assert a1 and a2
    assert a1 == a2

def test_slide_deck_vox_intersect_in_native():
    vox = [24, 22, 9]
    volume = Image(testsPath / 'testdata/volume_in_DTI')
    slide_deck  = Image(testsPath / 'testdata/slidedeck_in_PSOCT')
    psoct2dti = testsPath / 'testdata/PSOCT_to_DTI.mat'
    psoct2dti = readFlirt(psoct2dti)
    psoct2dti = fromFlirt(psoct2dti, slide_deck, volume, from_='world', to='world')
    slide = [Image(testsPath / 'testdata/slice_085_in_PSOCT'), Image(testsPath / 'testdata/slice_086_in_PSOCT')]
    a1 = cm.slide_vox_intersect(vox, volume, slide, psoct2dti)
    a2 = cm.slide_deck_vox_intersect(vox, volume, slide_deck, psoct2dti,direction="coronal")

    for sl in range(len(slide)):
        if a1[sl] is True:
            assert int(slide[sl].name.split('_')[1]) in a2
        else:
            assert not int(slide[sl].name.split('_')[1]) in a2

def test_slide_deck_vox_intersect_warp():
    vox = [24, 22, 9]
    volume = Image(testsPath / 'testdata/volume_in_DTI')
    slide_deck  = Image(testsPath / 'testdata/slidedeck_in_PSOCT')
    slide = [Image(testsPath / 'testdata/slice_085_in_PSOCT'), Image(testsPath / 'testdata/slice_086_in_PSOCT')]
    psoct2dti = testsPath / 'testdata/PSOCT_to_DTI_warpfield.nii.gz'
    psoct2dti = readFnirt(psoct2dti, slide_deck, volume)
    a1 = cm.slide_vox_intersect(vox, volume, slide, psoct2dti)
    a2 = cm.slide_deck_vox_intersect(vox, volume, slide_deck, psoct2dti,direction="coronal")

    for sl in range(len(slide)):
        if a1[sl] is True:
            assert int(slide[sl].name.split('_')[1]) in a2
        else:
            assert not int(slide[sl].name.split('_')[1]) in a2

def test_cube_vertices():
    a = cm.cube_vertices([0,0,0], 1)
    assert np.all(np.abs(a)==0.5)

def test_in_cube():
    coords = np.random.rand(50,3)
    pos    = [.5,.5,.5]
    size   = [1.2,1.2,1.2]
    a = cm.in_cube(coords, size, pos)
    assert np.all(a)

def test_get_pixgrid():
    slide = Image(testsPath / 'testdata/slice2')
    volume = Image(testsPath / 'testdata/volume')
    voxel = [25, 28, 9]
    a,b = cm.get_pixgrid(voxel, volume, slide, direction="coronal")
    assert len(a) > 0
    assert len(a) == len(b)

def test_get_pixgrid_in_native():
    voxel = [24, 22, 9]
    volume = Image(testsPath / 'testdata/volume_in_DTI')
    slide = Image(testsPath / 'testdata/slice_085_in_DTI')
    a1,b1 = cm.get_pixgrid(voxel, volume, slide)
    slide_deck  = Image(testsPath / 'testdata/slidedeck_in_PSOCT')
    psoct2dti = testsPath / 'testdata/PSOCT_to_DTI.mat'
    psoct2dti = readFlirt(psoct2dti)
    psoct2dti = fromFlirt(psoct2dti, slide_deck, volume, from_='world', to='world')
    slide = Image(testsPath / 'testdata/slice_085_in_PSOCT')
    a2,b2 = cm.get_pixgrid(voxel, volume, slide, psoct2dti, direction="coronal")
    a3,b3 = cm.get_pixgrid(voxel, volume, slide, direction="coronal")

    assert len(a1) > 0 and len(a2) > 0
    assert len(a1) == len(b1) == len(a2) == len(b2)
    assert np.allclose(a1, a2)
    assert np.allclose(b1, b2)
    assert len(a2) != len(a3) or not np.allclose(a2, a3)
    assert len(b2) != len(b3) or not np.allclose(b2, b3)

def test_get_pixgrid_warp():
    voxel = [24, 22, 9]
    volume = Image(testsPath / 'testdata/volume_in_DTI')
    slide_deck  = Image(testsPath / 'testdata/slidedeck_in_PSOCT')
    psoct2dti = testsPath / 'testdata/PSOCT_to_DTI.mat'
    psoct2dti = readFlirt(psoct2dti)
    psoct2dti = fromFlirt(psoct2dti, slide_deck, volume, from_='world', to='world')
    slide = Image(testsPath / 'testdata/slice_085_in_PSOCT')
    a1,b1 = cm.get_pixgrid(voxel, volume, slide, psoct2dti, direction="coronal")
    psoct2dti = testsPath / 'testdata/PSOCT_to_DTI_warpfield.nii.gz'
    dti2psoct = testsPath / 'testdata/DTI_to_PSOCT_warpfield.nii.gz'
    psoct2dti = readFnirt(psoct2dti, slide_deck, volume)
    dti2psoct = readFnirt(dti2psoct, volume, slide_deck)
    a2,b2 = cm.get_pixgrid(voxel, volume, slide, psoct2dti, dti2psoct, direction="coronal")

    assert len(a1) > 0 and len(a2) > 0
    # because arrays may have different lengths, we test the means only
    assert np.allclose(np.array(a1).mean(axis=0), np.array(a2).mean(axis=0), atol=5)
    assert np.allclose(np.array(b1).mean(axis=0), np.array(b2).mean(axis=0), atol=5)

# def test_slide_to_volume():
#     res = np.array([[ 0.06666344, -0.00106269, -0.00335597],
#                     [-0.00043999,  0.06583565, -0.01132557],
#                     [ 0.00048634,  0.01043948,  0.06561184]])

#     volume = Image(testsPath / 'testdata/volume')
#     slide  = Image(testsPath / 'testdata/slice1')

#     assert np.allclose(cm.slide_to_volume(volume, slide), res)

# def test_slide_to_volume_in_native():
#     volume = Image(testsPath / 'testdata/volume_in_DTI')
#     slide = Image(testsPath / 'testdata/slice_085_in_DTI')
#     a1 = cm.slide_to_volume(volume, slide)
#     slide_deck  = Image(testsPath / 'testdata/slidedeck_in_PSOCT')
#     psoct2dti = testsPath / 'testdata/PSOCT_to_DTI.mat'
#     slide = Image(testsPath / 'testdata/slice_085_in_PSOCT')
#     a2 = cm.slide_to_volume(volume, slide, slide_deck, psoct2dti)
#     a3 = cm.slide_to_volume(volume, slide)

#     assert np.allclose(a1, a2)
#     assert len(a2) != len(a3) or not np.allclose(a2, a3)

# def test_angle_to_vector():
#     a = np.arange(-180, 180+45, 45)*np.pi/180.0
#     assert cm.angle_to_vector(a).shape[-1] == 3
#     xform = np.eye(3)
#     assert cm.angle_to_vector(a, xform=xform).shape[-1] == 3

def test_warpfield_use():
    from fsl.transform.fnirt import readFnirt
    from fsl.transform.affine import transform, invert

    psoct2dti = testsPath / 'testdata/PSOCT_to_DTI_warpfield.nii.gz'
    dti2psoct = testsPath / 'testdata/DTI_to_PSOCT_warpfield.nii.gz'
    volume = Image(testsPath / 'testdata/volume_in_DTI')
    slide_deck  = Image(testsPath / 'testdata/slidedeck_in_PSOCT')
    slide = Image(testsPath / 'testdata/slice_085_in_PSOCT')
    
    vox = [24, 22, 9]
    # convert voxel to slide_deck world space
    psoct2dti = readFnirt(psoct2dti, Image(slide_deck), Image(volume))
    vox_in_world = psoct2dti.transform(vox, 'voxel', 'world')
    # convert slide_deck world space to slide voxel space
    # voxel spaces of slide vs slide_deck differ, but world spaces are the same
    world2pix = slide.getAffine('world', 'voxel')
    pix = transform(vox_in_world, world2pix)

    # convert pixel back to slide_deck/slide world space
    pix2world = transform(pix, invert(world2pix))
    # convert slide_deck world space back to volume voxel space
    dti2psoct = readFnirt(dti2psoct, Image(volume), Image(slide_deck))
    new_vox = dti2psoct.transform(pix2world, 'world', 'voxel')

    assert np.allclose(vox, new_vox, atol=1)

