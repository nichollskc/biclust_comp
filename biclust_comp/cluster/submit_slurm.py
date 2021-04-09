"""Submit a job script to slurm, adding required arguments.
Run snakemake using slurm with the command:
`snakemake --cluster "python biclust_comp/cluster/submit_slurm.py --account=ACCOUNT_NAME" \
--jobs 100 --cluster-config biclust_comp/cluster/slurm_config.json`

Or, to submit all jobs at once, with dependencies set up:
`snakemake --cluster "python biclust_comp/cluster/submit_slurm.py {dependencies} --account=ACCOUNT_NAME" \
--immediate-submit --notemp --jobs 100 --cluster-config biclust_comp/cluster/slurm_config.json`
"""
#!python3
from pathlib import Path
import re
import subprocess
import sys
import time

from snakemake.utils import read_job_properties

# Check that the log directory exists
Path("logs/cluster").mkdir(parents=True, exist_ok=True)

# Job script will be last argument
jobscript = sys.argv[-1]

# Second last argument will be the account argument
account = sys.argv[-2]

# Check that the job script argument really is a jobscript
# It should contain 'snakejob' and be a script
assert re.match(r'\S+/snakejob\.\S+\.sh', jobscript), \
       f"Job script name {jobscript} doesn't match expected form"

# Check account argument is of the form --account=ACCOUNT_NAME
assert re.match(r'^--account=\S+$', account), \
       f"Account argument name {account} doesn't match expected form, should be " \
       f"'--account=ACCOUNT_NAME'"

# Read properties from the header of jobscript using snakemake utility function
job = read_job_properties(jobscript)

if 'rule' in job:
    job_name = job['rule']
else:
    job_name = job['groupid']
info_str = f"JOBNAME: {job_name}"

try:
    first_input = job['input'][0]
    info_str += f" INPUT1: {first_input}"
except (KeyError, IndexError) as e:
    # Job has no input, so don't add input file to log
    pass

if 'output' in job:
    first_output = job['output'][0]
    info_str += f" OUTPUT1: {first_output}"

command = f"sbatch " \
          f"--output=logs/cluster/{job_name}_{job['jobid']}_slurm_%j.out " \
          f"--error=logs/cluster/{job_name}_{job['jobid']}_slurm_%j.err " \
          f"--nodes={job['cluster']['nodes']} " \
          f"--ntasks={job['cluster']['cores']} " \
          f"--time={job['cluster']['max_time']} " \
          f"--partition={job['cluster']['queue']} " \
          f"{account} " \
          f"--parsable "

# Any remaining arguments will be slurm IDs of jobs upon which this job is
# dependent
dependencies = set(sys.argv[1:-2])
if dependencies:
    command += f"--dependency=afterok:{','.join(dependencies)} "

# Add the job script itself to the command
command += jobscript

# Run the command, capturing the slurm_id so we can save it to a log file
result = subprocess.run(command.split(' '), stdout=subprocess.PIPE)
slurm_id = result.stdout.decode('utf-8').strip()
timestamp = time.strftime('%Y-%m-%d_%H:%M:%S')

with open("logs/sbatch.log", 'a') as f:
    f.write(f"TIME: {timestamp} SLURM_ID: {slurm_id} {info_str}\n")

# Print the ID so that snakemake knows it has been submitted
print(slurm_id)

