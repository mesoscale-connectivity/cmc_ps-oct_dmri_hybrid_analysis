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
cmc_hybrid  --ori_slides_dir <PSOCT_ORIENTATION_FOLDER> \
            --slide_mapping <PSOCT_FOLDER>/slidedeck_slice_mapping.json \
            --bpx <BEDPOSTX_FOLDER>> \
            -o <OUTPUT_FILE> \
            --mask <BRAIN_MASK_IMAGE>  \
            -s 0.4 0.4 0.4 \
            --roi <xmin> <xsize> <ymin> <ysize> <zmin> <zsize> \
            --slide2vol=<PSOCT_FOLDER>/PSOCT_to_MRI_warpfield.nii.gz \
            --vol2slide=<PSOCT_FOLDER>/MRI_to_PSOCT_warpfield.nii.gz \
            --slidedeck=<PSOCT_FOLDER>/Ori_slide_deck.nii.gz \
            --verbose 
	   
```

## Using the code within python

MORE INSTRUCTIONS TBD
(see Notebook)

