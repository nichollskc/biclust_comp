import argparse
import re
import subprocess

import junit_xml

def main(exit_code, logfile, command, test_name):
    test_cases = []

    # If there was a failure, we need to report it
    if exit_code:
        grep_command = f"grep -A10 -e 'Error in rule .*' {logfile}"
        output = subprocess.getoutput(grep_command)
        rule_errors = output.split('\n--\n')

        for index, rule_error in enumerate(rule_errors):
            rule_name = f"test_name_{index}"

            match = re.match('Error in rule (.*):', rule_error)
            if match:
                rule_name = match[1]

            test_case = junit_xml.TestCase(rule_name, log=logfile)
            test_case.add_error_info(message=f"Error in snakemake run using command '{command}'",
                                     output=rule_error)

            test_cases.append(test_case)
    else:
        test_case = junit_xml.TestCase(test_name, log=logfile)
        test_cases.append(test_case)

    test_suite = junit_xml.TestSuite(test_name, test_cases)

    test_output_filename = f"tests/test-results/{test_name}/results.xml"
    with open(test_output_filename, 'w') as f:
        junit_xml.TestSuite.to_file(f, [test_suite], prettyprint=True)
        print(f"Saved XML test results to {test_output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make an xml report of the test consisting of running "
                                                 "a given snakemake command. Requires the log file and "
                                                 "exit code.")
    required_named = parser.add_argument_group('required named arguments')
    required_named.add_argument("--exit_code",
                                required=True,
                                help="exit code that snakemake gave",
                                type=int)
    required_named.add_argument("--logfile",
                                required=True,
                                help="snakemake logfile path")
    required_named.add_argument('--snakemake_command',
                                required=True,
                                help="full command given to snakemake")
    required_named.add_argument('--test_name',
                                required=True,
                                help="name to give this snakemake test")

    args = parser.parse_args()
    main(args.exit_code, args.logfile, args.snakemake_command, args.test_name)
