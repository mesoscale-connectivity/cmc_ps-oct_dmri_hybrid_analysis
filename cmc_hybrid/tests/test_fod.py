
import numpy as np
from cmc_hybrid import fod

def test_form_SHmat():
    xyz = np.random.randn(50,3)
    SH = fod.form_SHmat(xyz, coord_system='cart')
    assert SH.shape == (50, 45)


def test_hybrid_vecs():
    th = np.random.randn(100)
    ph = np.random.randn(100)
    f = np.random.rand(100)
    vecs = np.random.randn(200,3)
    new_vecs = fod.hybrid_vecs(th, ph, f, vecs)
    assert new_vecs.shape == (200,3)

    #
    # vecs = np.random.randn(200,3)
    # vecs[:,1] = 0.
    #
    # th = np.ones(100) * np.pi/2.
    # ph = np.random.randn(100)
    # f = np.random.rand(100)
    #
    # new_vecs = fod.hybrid_vecs(th, ph, f, vecs)
    #
    # assert np.all(np.isclose(new_vecs[:,1],0.))





