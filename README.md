# Biclustering Comparison

Code to compare a number of biclustering methods on real gene expression datasets and synthetic datasets, as well as to perform stability testing.

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
## Table of Contents

- [Methods](#methods)
- [Installation](#installation)
    - [Conda environment](#conda-environment)
    - [Matlab setup](#matlab-setup)
    - [Packages not in conda](#packages-not-in-conda)
        - [Issues installing BicMix](#issues-installing-bicmix)
- [Usage](#usage)
- [Tests](#tests)
    - [CircleCI](#circleci)
- [Processing real datasets](#processing-real-datasets)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->
*(Table of contents generated with [DocToc](https://github.com/thlorenz/doctoc): `doctoc README.md`)*

## Methods

* BicMix (Bayesian biclustering)
* FABIA
* MultiCluster
* nsNMF (non-smooth Non-negative matrix factorisation)
* Plaid
* SDA (Sparse Decomposition of Arrays) - executable not licensed for distribution
* SNMF (Sparse Non-negative matrix factorisation)
* SSLB (Spike-and-slab Lasso biclustering)

Most of the methods factorise the data matrix as a product of two sparse matrices. SDA and MultiCluster are exceptions to this, as they factorise it into a tensor product of three sparse matrices. Another exception is SNMF which only enforces sparsity on one of the matrices.

## Installation

Clone the repository:

```
git clone https://github.com/nichollskc/biclustering_comparison.git
```

Running the command below should be sufficient to set up matlab (if it is available - only required for MultiCluster) and conda environment (if conda is installed), and to install the few packages not available in conda.

```
source install_dependencies.sh
```

For more details of the individual steps, see the sections below.

## Availability of SDA

SDA is not licensed for distribution, so is not included in this repository. See the [SDA website](https://jmarchini.org/sda/) for information about accessing the executable, and then place the sda folder in the root of the repository so that the scripts can call the executable (in [run_biclustering.rules](biclust_comp/workflows/run_biclustering.rules)).

#### Conda environment

To install conda, follow the instructions on the [conda website](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).

Use the environment file provided to create the environment and then activate it.

```
conda env create -f environment-minimal.yml -n biclustering_comparison
conda activate biclustering_comparison
```

#### Matlab setup

Matlab is only required for the MultiCluster method.

Ensure that `which matlab` finds the matlab installation. You may need to add the location of the matlab executable to the path.

Install the tensor Toolbox:

```
wget https://gitlab.com/tensors/tensor_toolbox/-/archive/v3.1/tensor_toolbox-v3.1.zip
unzip tensor_toolbox-v3.1.zip
```

#### Packages not in conda

[Nimfa](http://nimfa.biolab.si/) is required for the two non-negative matrix factorisation methods (SNMF, nsNMF). It can be installed using `pip`:

```
pip install nimfa
```

The SSLB method is available by running `remotes::install_github('gemma-e-moran/SSLB')` in R. The following command runs this from the command line.

```
Rscript -e "remotes::install_github('gemma-e-moran/SSLB')"
```

BicMix is available either as a standalone C++ package, or as an R wrapper around the C++ code. We use the R version, which can be installed using similar syntax to the above `remotes::install_github('chuangao/BicMix')`, or alternatively the following:

```
git clone https://github.com/chuangao/BicMix
R CMD INSTALL BicMix
```

###### Issues installing BicMix

I have faced issues installing BicMix and found that changing the parameters in ~/.R/Makevars was crucial. This is the ~/.R/Makevars contents that worked on the HPC:

```
CC=gcc
CXX=g++
CXX11=g++
```

## Usage

Ensure the conda environment installed in [installation](#installation) has been loaded:

```
conda activate biclustering_comparison
```

This project consists of a snakemake pipeline which runs the methods on a range of datasets and compiles the results. This can be run with the following command:

```
snakemake
```

## Tests

#### CircleCI

By default every commit that is pushed is tested with CircleCI. The script [here](.circle/config.yml) describes the steps taken in the CircleCI builds. The steps taken are:

1. Load a docker image with miniconda loaded
2. Install the conda environment from environment-minimal.yml
3. Install other required packages
4. Save environment to a cache so it can be loaded quickly on subsequent builds if the requirements haven't changed
5. Run snakemake test pipeline
6. Run tests using nosetests

CircleCI is not set up to run Matlab scripts, so MultiCluster isn't run here. There was also an odd bug where BicMix would sometimes take 5 minutes to run something that usually takes only seconds. For that reason, BicMix is not run in the CircleCI workflow.

# Using as submodule

```
git submodule add https://github.com/nichollskc/biclustering_comparison.git
```

Then you can write your own Snakefile in the working directory, and include the Snakefile of the module using the snakemake `include` directive.

```
include: "biclustering_comparison/Snakefile"
```

You will need to include in a config file a line such as below, to help the biclust\_comp workflow find the scripts it needs (relative to the working directory).

```
BICLUST_COMP_SCRIPTS_RELATIVE: 'biclustering_comparison/'
```

# Processing real datasets

The code to generate IMPC tpm.tsv file is in the repo here https://github.com/nichollskc/IMPC_analysis
For the IMPC data, I copied the tpm.tsv file into the right place in this repo, then ran this snakemake pipeline.

The code to generate Benaroya/Presnell tpm.tsv file is here https://github.com/nichollskc/E-GEOD-60424
For this data, the github repo used this repo as a submodule.
