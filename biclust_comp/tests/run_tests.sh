if [ -z $1 ]
then
    OUTPUTDIR=tests/test-results/default/$(date +%F%T)
else
    GIT_BRANCH=$1
    OUTPUTDIR=tests/test-results/${GIT_BRANCH}/$(date +%F%T)
    echo "Running tests on branch ${GIT_BRANCH}"

    # Change branch
    git fetch
    git checkout $GIT_BRANCH
    git pull
    git status
fi

mkdir -p $OUTPUTDIR

# Remove previous results
rm -rf results/ data/ analysis/

# Set up environment - use a special conda environment for testing
BIC_COMP_CONDA_ENV='bic_comp_testing'
# Use source so that the script can set environment variables and activate environment
source ./install_dependencies.sh

# Copy files we need for IMPC analysis
mkdir -p data/real/IMPC/raw
mkdir -p analysis
cp /rds/user/kcn25/hpc-work/biclustering_comparison/analysis/mart_export.txt analysis/mart_export.txt
cp /rds/user/kcn25/hpc-work/IMPC/data/tpm.tsv data/real/IMPC/tpm_full.tsv
cut -f -1000 data/real/IMPC/tpm_full.tsv > data/real/IMPC/tpm.tsv
cp /rds/user/kcn25/hpc-work/IMPC/sample_info.txt data/real/IMPC/raw/sample_info.txt

# Run snakemake - do dry run, dry run of test and finally a full test run
# In each case, capture the return code using PIPESTATUS
snakemake --dry-run 2>&1 | tee $OUTPUTDIR/snakemake_dryrun.log
SNAKEMAKE_DRY_RC=${PIPESTATUS[0]}

snakemake --configfile biclust_comp/workflows/snakemake_testing_config.yml --dry-run test_generate_results 2>&1 | tee $OUTPUTDIR/snakemake_dryrun_test_generate.log
SNAKEMAKE_DRY_TEST_GENERATE_RC=${PIPESTATUS[0]}
snakemake --configfile biclust_comp/workflows/snakemake_testing_config.yml --keep-going test_generate_results 2>&1 | tee $OUTPUTDIR/snakemake_test_generate.log
SNAKEMAKE_TEST_GENERATE_RC=${PIPESTATUS[0]}

snakemake --configfile biclust_comp/workflows/snakemake_testing_config.yml --dry-run test 2>&1 | tee $OUTPUTDIR/snakemake_dryrun_test.log
SNAKEMAKE_DRY_TEST_RC=${PIPESTATUS[0]}
snakemake --configfile biclust_comp/workflows/snakemake_testing_config.yml --keep-going test 2>&1 | tee $OUTPUTDIR/snakemake_test.log
SNAKEMAKE_TEST_RC=${PIPESTATUS[0]}

# Run nosetests
nosetests biclust_comp/tests --cover-package biclust_comp --cover-xml --with-xunit --xunit-file=$OUTPUTDIR/nosetest_results.xml --verbose 2>&1 | tee $OUTPUTDIR/nosetests.log
NOSETEST_RC=${PIPESTATUS[0]}

echo snakemake dry run: $SNAKEMAKE_DRY_RC
echo snakemake test_generate dry run: $SNAKEMAKE_DRY_TEST_GENERATE_RC
echo snakemake test_generate run: $SNAKEMAKE_TEST_GENERATE_RC
echo snakemake test dry run: $SNAKEMAKE_DRY_TEST_RC
echo snakemake test run: $SNAKEMAKE_TEST_RC
echo nosetest: $NOSETEST_RC

# Exit with 1 if any of the tests failed, and 0 otherwise
! (( $SNAKEMAKE_DRY_RC || $SNAKEMAKE_DRY_TEST_GENERATE_RC || $SNAKEMAKE_TEST_GENERATE_RC || $SNAKEMAKE_DRY_TEST_RC || $SNAKEMAKE_TEST_RC || $NOSETEST_RC ))
