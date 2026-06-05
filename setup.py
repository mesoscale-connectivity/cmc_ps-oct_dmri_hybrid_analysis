#!/usr/bin/env python

from setuptools import setup

with open('requirements.txt', 'rt') as f:
    install_requires = [line.strip() for line in f.readlines()]

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='cmc_hybrid',
    description='Joint Modelling of dMRI and Microscopy (PSOCT)',
    version="0.2.1",

    author=['Saad Jbabdi', 'Silei Zhu', 'Amy Howard'],
    author_email='<saad.jbabdi@ndcn.ox.ac.uk> & <silei.zhu@ndcn.ox.ac.uk> & <amy.howard@ndcn.ox.ac.uk>',
    long_description=long_description,
    long_description_content_type="text/markdown",

    packages=['cmc_hybrid',],
    install_requires=install_requires,
    extras_require={
        "dev": ["pytest",
                "flake8"],
    },

    scripts=['cmc_hybrid/scripts/cmc_hybrid.py',
             'cmc_hybrid/scripts/cmc_slice_mask.py',
             'cmc_hybrid/scripts/cmc_bpx2fod.py']
)
