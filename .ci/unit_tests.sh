eval "$(micromamba shell hook --shell bash)"

set -e

projroot=$(cd $(dirname $0) && pwd)/..
micromamba create -y \
  -c https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/ \
  -c conda-forge \
  -p ./fsl \
  fsl-avwutils fsl-flirt \
  python $(cat ${projroot}/requirements.txt)
micromamba activate ./fsl
pip install .
pytest -v .