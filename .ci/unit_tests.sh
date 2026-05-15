eval "$(micromamba shell hook --shell bash)"

set -e

projroot=$(cd $(dirname $0) && pwd)/..
micromamba create -y \
  -c https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/ \
  -c conda-forge \
  -p ./fsl \
  fsl-avwutils fsl-flirt git \
  python
micromamba activate ./fsl

pip install --upgrade pip
pip install -r "${projroot}/requirements.txt"
pip install .[dev]

# Make sure we have latest (possibly development)
# versions of the core dependencies
PIPARGS=" --retries 10 "
PIPARGS+="--timeout 30 "
PIPARGS+="--trusted-host files.pythonhosted.org "
PIPARGS+="--trusted-host pypi.org "

pip install $PIPARGS git+https://git.fmrib.ox.ac.uk/fsl/fslpy.git

export FSLDIR=./fsl
source $FSLDIR/etc/fslconf/fsl.sh
pytest -v .
