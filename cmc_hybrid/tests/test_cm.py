from cmc_hybrid import coordinate_mapping as cm
from pathlib import Path
testsPath = Path(__file__).parent
import numpy as np
from fsl.data.image import Image

def test_vox_to_pix():
    volume = Image(testsPath / 'testdata/volume')
    slices = [Image(x) for x in [testsPath / 'testdata/slice1', testsPath / 'testdata/slice2']]
    vox    = [25, 28, 9]
    a,b,c  = cm.vox_to_pix(vox, slices, volume)
    assert len(a)>0

def test_slide_vox_intersect():

    slide = Image(testsPath / 'testdata/slice2')
    volume= Image(testsPath / 'testdata/volume')
    a = cm.slide_vox_intersect([25, 28, 9], slide, volume)
    assert a

    a = cm.slide_vox_intersect([25, 10, 9], slide, volume)
    assert not a

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
    a,b = cm.get_pixgrid(voxel, slide, volume)
    assert len(a) > 0
    assert len(a) == len(b)

def test_slide_to_volume():
    res = np.array([[ 0.06666344, -0.00106269, -0.00335597],
                    [-0.00043999,  0.06583565, -0.01132557],
                    [ 0.00048634,  0.01043948,  0.06561184]])

    volume = Image(testsPath / 'testdata/volume')
    slide  = Image(testsPath / 'testdata/slice1')

    assert np.all(np.isclose(cm.slide_to_volume(slide, volume), res))


def test_angle_to_vector():
    a = np.arange(-180, 180+45, 45)*np.pi/180.0
    assert cm.angle_to_vector(a).shape[-1] == 3
    xform = np.eye(3)
    assert cm.angle_to_vector(a, xform=xform).shape[-1] == 3
