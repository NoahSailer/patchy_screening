#!/bin/zsh
set -e
ENV_NAME="patchy"
__conda_setup="$($(which conda) 'shell.zsh' 'hook' 2> /dev/null)"
eval "$__conda_setup"
# check if patchy environment exists
if conda env list | grep -qE "^$ENV_NAME\s"; then
    echo "Conda environment '$ENV_NAME' exists. Removing it..."
    conda remove --name "$ENV_NAME" --all -y
    echo "Environment '$ENV_NAME' removed."
else
    echo "Conda environment '$ENV_NAME' does not exist. Proceeding..."
fi
# clone base environment
echo "Cloning 'base' environment to '$ENV_NAME'..."
conda create --name "$ENV_NAME" --clone base -y
conda activate "$ENV_NAME"
echo "Environment '$ENV_NAME' created successfully. Proceeding..."
# install some relevant packages and set up jupyter
pip install pyfftw healpy classy pixell jupyter ipykernel ipython --user
python -m ipykernel install --user --name "$ENV_NAME" --display-name "$ENV_NAME"