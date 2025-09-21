# CMC Hybrid FOD Modelling

Tools for combining dMRI and microscopy to create hybrid FOD models.


## Getting started
Create a conda environment (this is not strictly necessary but can help keep your environments clean):

```commandline
conda create -n hybrid python==3.10
conda activate hybrid
```

Clone repo and install:
```commandline
git clone https://git.fmrib.ox.ac.uk/saad/cmc_hybrid.git
cd cmc_hybrid
pip install .
```

## Running the wrapper script
To get help with the script, type e.g.:

```bash
cmc_hybrid --help
```

Here is an example call to the main script:

```bash
cmc_hybrid --psoct <PSOCT_FOLDER>/Slices*_header.nii.gz \
	       --bpx <BEDPOSTX_FOLDER>> \
	       -o <OUTPUT_FILE> \
	       --mask <BRAIN_MASK_IMAGE  \
	       --scale 8 --roi <xmin> <xsize> <ymin> <ysize> <zmin> <zsize> \
	       --verbose 
	   
```


MORE INSTRUCTIONS TBD

