
import numpy as np
from cmc_hybrid import fod

def test_form_SHmat():
    xyz = np.random.randn(50,3)
    SH = fod.form_SHmat(xyz, coord_system='cart')
    assert SH.shape == (50, 45)

    th = [0., np.pi/4., np.pi/2.]
    ph = [0., np.pi/4., 0. ]
    SHmat = fod.form_SHmat([th,ph], max_order=4, coord_system='polar')
    assert np.isclose(np.mean(SHmat), 0.056381724340180316)


def test_sample_from():
    dens = np.array([[.1,.2,.7],
                 [.2,.2,.2],
                 [.6,.3,.1]])
    samples = []
    for _ in range(10000):
        samples.append(fod.sample_from( dens, axis=1 ))
    samples = np.array(samples)
    hist = np.histogram(samples[:,0], bins=(np.arange(3+1)-0.5), density=True)
    assert np.all(np.isclose(hist[0], [.1,.2,.7], atol=1e-1))


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


def test_SphereKernelDensity():
    N = 100
    sig = 0.1
    z = np.ones(N)
    x = np.random.randn(N) * sig
    y = np.random.randn(N) * sig
    xyz1 = np.stack([x,y,z], axis=1)
    xyz2 = np.stack([z,x,y], axis=1)
    skde = fod.SphereKernelDensity(bandwidth=20.)
    skde.fit(xyz1)
    assert skde.pdf(xyz1).mean() > skde.pdf(xyz2).mean()


def test_fit_sh_fod():
    N = 100
    sig = 0.1
    z = np.ones(N)
    x = np.random.randn(N) * sig
    y = np.random.randn(N) * sig
    xyz = np.stack([x,y,z], axis=1)
    from cmc_hybrid import utils
    xyz = utils.vec_normalise(xyz, axis=1)
    coeffs = fod.fit_sh_fod(xyz, max_order=8)
    assert len(coeffs) == 45


def test_plot_odf_glyph():
    N = 100
    sig = 0.1
    z = np.ones(N)
    x = np.random.randn(N) * sig
    y = np.random.randn(N) * sig
    xyz1 = np.stack([x,y,z], axis=1)
    xyz2 = np.stack([z,x,y], axis=1)

    skde = fod.SphereKernelDensity(bandwidth=20.)
    skde.fit(xyz1)

    fig = fod.plot_odf_glyph(skde, samples=xyz1)

    import plotly
    assert type(fig) == plotly.graph_objs._figure.Figure




