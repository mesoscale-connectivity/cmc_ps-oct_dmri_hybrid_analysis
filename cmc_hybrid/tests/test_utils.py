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
    print(v)
    assert np.all(np.isclose(v,np.array([1,0,0])))

def test_prepare_mask():
    brainmask = Image(testsPath / 'testdata/volume').data
    maskfile  = testsPath / 'testdata/volume'
    roi = (0, -1, 0, -1, 0, -1)
    a = utils.prepare_mask(maskfile, roi)
    assert np.all(a.data == brainmask)

    a = utils.prepare_mask(maskfile, roi, scale=2)
    assert a.shape[0] == 2*Image(maskfile).shape[0]

def test_fudge_psoct_orientation():
    res = np.array([ 90. ,  67.5,  45. ,  22.5, 180. , 157.5, 135. , 112.5,  90. ])
    x = utils.fudge_psoct_orientation( np.arange(-180, 180+45, 45)*np.pi/180.0, 0. ) * 180.0 / np.pi
    assert np.all(np.isclose(x, res))


    theta = [0., 45., 180.]
    x = utils.fudge_psoct_orientation( theta )
    assert len(x) == len(theta)

def test_upscale_image():
    from fsl.data.image import Image
    img = Image(testsPath / 'testdata/volume')
    scale = 2
    newimg = utils.upscale_image(img, scale)
    assert np.all(np.array(newimg.pixdim)==1.5)

def test_vec_normalise():
    x = np.random.randn(100,20)
    n = np.sum((utils.vec_normalise(x, axis=0))**2, axis=0)
    assert np.all(np.isclose(n,1.))
    n = np.sum((utils.vec_normalise(x, axis=1))**2, axis=1)
    assert np.all(np.isclose(n,1.))
