from cmc_hybrid import utils
from pathlib import Path
testsPath = Path(__file__).parent
import numpy as np
from fsl.data.image import Image

def test_dirgen():
    mat = utils.dirgen(128)
    assert mat.shape == (128, 3)


def test_pol2cart():
    th = [0., np.pi/2.]
    ph = [0., np.pi]
    x = utils.pol2cart(th,ph)
    assert len(x) == len(th)
    assert np.all(np.isclose(x[0], np.array([0,0,1])))
    assert np.all(np.isclose(x[1], np.array([-1,0,0])))

def test_cart2pol():
    xyz = [[0,0,1],[-1,0,0]]
    th, ph = utils.cart2pol(xyz)
    assert np.all(np.isclose(th, [0., np.pi/2.]))
    assert np.all(np.isclose(ph, [0., np.pi]))

def test_make_dyads():
    vecs = [[1,0,0],[1,0,0],[1,0,0]]
    v = utils.make_dyads(vecs)
    assert np.all(np.isclose(v,np.array([1,0,0])))

# def test_prepare_mask():
#     brainmask = Image(testsPath / 'testdata/volume').data
#     maskfile  = testsPath / 'testdata/volume'
#     roi = (0, -1, 0, -1, 0, -1)
#     a = utils.prepare_mask(maskfile, roi, resolution=[0.4,0.4,0.4])
#     assert np.all(a.data == brainmask)

#     a = utils.prepare_mask(maskfile, roi, resolution=[0.2,0.2,0.2])
#     assert a.shape[0] == 2*Image(maskfile).shape[0]

#     slides = [Image(testsPath / 'testdata/slice1'), Image(testsPath / 'testdata/slice2')]
#     a = utils.prepare_mask(maskfile, roi, resolution=[0.2,0.2,0.2], slides=slides)
#     assert np.sum(a.data) > 0

def test_prepare_mask():
    from fsl.data.image import Image
    maskfile  = testsPath / 'testdata/volume'
    img = Image(maskfile)
    roi = (0, -1, 0, -1, 0, -1)
    
    # Get the ACTUAL original pixdim and shape
    orig_pixdim = img.pixdim[0]
    orig_shape = img.shape[0]

    # Test 1: Use the actual original resolution (so it doesn't scale)
    a = utils.prepare_mask(maskfile, roi=roi, resolution=[orig_pixdim]*3)
    assert a.shape[0] == orig_shape

    # Test 2: Scale to 0.2mm
    target_res = 0.2
    a_high = utils.prepare_mask(maskfile, roi=roi, resolution=[target_res]*3)
    
    # Calculate expected shape: (Original Size * Original Res) / Target Res
    # We use round() because shapes must be integers
    expected_shape = int(round(orig_shape * (orig_pixdim / target_res)))
    
    assert a_high.shape[0] == expected_shape

def test_order_voxels():
    v = np.random.randn(10,3)
    vv = utils.order_voxels(v, 'Coronal')
    assert vv[0,1] <= vv[-1,1]
    vv = utils.order_voxels(v, 'Sagittal')
    assert vv[0,0] <= vv[-1,0]
    vv = utils.order_voxels(v, 'Axial')
    assert vv[0,2] <= vv[-1,2]


def test_fudge_psoct_orientation():
    res = np.array([90.0, 67.5, 45.0, 22.5, 180.0, 157.5, 135.0, 112.5, 90.0])
    
    input_angles = np.arange(-180, 180+45, 45) * np.pi / 180.0
    x = utils.fudge_psoct_orientation(input_angles, 0.) * 180.0 / np.pi
    
    assert np.all(np.isclose(x, res))

    theta = [0., 45., 180.]
    x = utils.fudge_psoct_orientation(np.array(theta) * np.pi / 180.0, 0.)
    assert len(x) == len(theta)

def test_upscale_image():
    from fsl.data.image import Image
    # 1. Load the original image
    img = Image(testsPath / 'testdata/volume')
    target = (1.5, 1.5, 1.5)

    newimg = utils.upscale_image(img, target_pixdims=target)
    assert np.all(np.isclose(newimg.pixdim[:3], target))

def test_vec_normalise():
    x = np.random.randn(100,20)
    n = np.sum((utils.vec_normalise(x, axis=0))**2, axis=0)
    assert np.all(np.isclose(n,1.))
    n = np.sum((utils.vec_normalise(x, axis=1))**2, axis=1)
    assert np.all(np.isclose(n,1.))

def test_get_data():
    maskfile  = testsPath / 'testdata/volume'
    maskdata  = utils.get_data(Image(maskfile))
    assert np.sum(maskdata) > 0

def test_slide_is_too_big():
    img = Image(testsPath / 'testdata/slice1')
    assert not utils.slide_is_too_big(img)


def test_resample_slide():
    img = Image(testsPath / 'testdata/slice1')
    img_r = utils.resample_slide(img, slide_direction='coronal', factor=2)
    assert img_r.pixdim[0] == img.pixdim[0]*2
    assert img_r.pixdim[2] == img.pixdim[2]*2
    assert img_r.pixdim[1] == img.pixdim[1]

