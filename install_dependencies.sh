#!/bin/bash
# Usage: BIC_COMP_CONDA_ENV='bic_comp' source install_dependencies.sh

# Add scripts directory to PYTHONPATH
#   (not ideal - would be neater to install my scripts as a package)
# This appends the current directory to the PYTHONPATH, adding a colon if the
#   PYTHONPATH is non-empty.
export "PYTHONPATH=${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"
echo "Added scripts directory to PYTHONPATH: ${PYTHONPATH}"

# See if a name for the conda environment has been passed
if [ -z "$BIC_COMP_CONDA_ENV" ]
  then
    BIC_COMP_CONDA_ENV='biclustering_comparison'
    echo "No environment name supplied, using default of '$BIC_COMP_CONDA_ENV'"
fi

# Check if Matlab is available (i.e. does `which matlab` succeed or not?)
if [[ $(which matlab) ]]
then
    # If Matlab is available, check if the required tensor toolbox is present
    if [[ ! $(ls tensor_toolbox-v3.1) ]]
    then
        # If not, download and install it
        wget https://gitlab.com/tensors/tensor_toolbox/-/archive/v3.1/tensor_toolbox-v3.1.zip
        unzip tensor_toolbox-v3.1.zip
    else
        echo "Tensor toolbox already present"
    fi
    # Add biclust_comp/methods and the tensor toolbox to the matlab path
    export MATLABPATH=$(pwd)/biclust_comp/methods:$(pwd)/tensor_toolbox-v3.1
    echo "Matlab has been set up"
else
    echo "Matlab will not be used"
fi

# Ensure conda activate command is available
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

# Try to activate environment - use 'cmd || echo' format to avoid CircleCI build failing
# if the environment doesn't exist - CircleCI runs in -e mode so any failing line causes
# the script to fail
echo "Attempting to activate environment $BIC_COMP_CONDA_ENV"
unset BIC_COMP_ACTIVATE_RETVAL
conda activate $BIC_COMP_CONDA_ENV || { BIC_COMP_ACTIVATE_RETVAL=$?; \
                                        echo "'conda activate $BIC_COMP_CONDA_ENV' failed"; }

# Check if the conda activate command failed (i.e. does RETVAL variable exist?)
# If so, this should only be because the environment doesn't exist
if [[ -z "$BIC_COMP_ACTIVATE_RETVAL" ]]
then
    echo "Conda environment $BIC_COMP_CONDA_ENV activated. Setup finished"
else
    if [[ $(which mamba) ]]
    then
        CONDA_ALTERNATIVE="mamba"
    else
        CONDA_ALTERNATIVE="conda"
    fi
    echo "Conda environment $BIC_COMP_CONDA_ENV doesn't exist yet. Installing it using $CONDA_ALTERNATIVE"
    $CONDA_ALTERNATIVE env create -f environment-minimal.yml -n $BIC_COMP_CONDA_ENV
    echo "Installed environment $BIC_COMP_CONDA_ENV"

    conda activate $BIC_COMP_CONDA_ENV
    git clone https://github.com/nichollskc/pyfabia && cd pyfabia && git checkout 3c7826c7855d && cd ../
    pip install pyfabia/

    pip install nimfa
    pip install intermine
    Rscript -e "remotes::install_github('gemoran/SSLB@115921f6e72bc6e499')"
    git clone https://github.com/chuangao/BicMix && cd BicMix && git checkout ba1cfdfc50311d && cd ../
    R CMD INSTALL BicMix
    echo "Setup finished"
fi
