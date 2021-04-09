#!/usr/bin/bash

source ~/.bashrc
source /etc/profile.d/modules.sh

module load matlab
module list

LOGDIR=tests/test-results/$(date +%F%T)
mkdir -p $LOGDIR

# git fetch - uses ssh key at ~/.ssh/github_id_rsa
ssh-agent sh -c 'ssh-add ~/.ssh/github_id_rsa; git fetch git@github.com:nichollskc/biclustering_comparison.git +refs/heads/*:refs/remotes/origin/*'
git status
git for-each-ref --sort=-committerdate refs/remotes/origin --format='%(committerdate:unix) %(refname:strip=-1)' > tests/branch_dates.txt

DEADLINE=$(date --date='yesterday' +%s)
echo $DEADLINE

RESULTS_FILE=$LOGDIR/results.txt

while read -r COMMIT_DATE BRANCH_NAME <&3; do
    echo "Read in branch"
    echo $BRANCH_NAME
    echo $COMMIT_DATE

    if [ $COMMIT_DATE -ge $DEADLINE ];
    then
        git checkout $BRANCH_NAME
        git status
        git reset --hard origin/$BRANCH_NAME
        git status

        echo "Date is since yesterday"
        LOGFILE=$LOGDIR/${BRANCH_NAME}.log 
        biclust_comp/tests/run_tests.sh 2>&1 | tee $LOGFILE
        if [ ${PIPESTATUS[0]} -ne 0 ]
        then
            echo "Tests on branch $BRANCH_NAME failed" | tee -a $RESULTS_FILE
            mail -a $LOGFILE -s "Nightly tests on branch ${BRANCH_NAME} failed" kcn25@cam.ac.uk
        else
            echo "Tests on branch $BRANCH_NAME passed" | tee -a $RESULTS_FILE
            mail -a $LOGFILE -s "Nightly tests on branch ${BRANCH_NAME} passed" kcn25@cam.ac.uk
        fi
    else
        echo "Branch $BRANCH_NAME ignored - last updated $(date --date=@${COMMIT_DATE} +%F\ %T) which is before $(date --date=@${DEADLINE} +%F\ %T)" | tee -a $RESULTS_FILE
    fi 
    echo "Moving on to next branch"
done 3< tests/branch_dates.txt

mail -a $RESULTS_FILE -s "Nightly tests summary" kcn25@cam.ac.uk
