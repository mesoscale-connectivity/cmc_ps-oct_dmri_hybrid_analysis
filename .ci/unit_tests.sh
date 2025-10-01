set -e

source create \
        -c https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/ \
        -c conda-forge \
        -p /fsl \
        fsl-avwutils fsl-flirt \
        python pytest
conda activate /fsl
pip install . pytest
export FSLDIR=/fsl
source $FSLDIR/etc/fslconf/fsl.sh
pytest -v .
