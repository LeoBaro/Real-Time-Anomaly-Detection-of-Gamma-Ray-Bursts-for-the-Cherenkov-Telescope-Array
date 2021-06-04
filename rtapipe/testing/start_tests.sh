#!/bin/bash

if [[ -z "${DATA}" ]]; then
    echo "Please, export \$DATA"
else
    pushd ./

    TEST_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

    coverage_report_dir="$TEST_DIR/output/coverage_reports"
    mkdir -p "$coverage_report_dir"
    rm -f "$coverage_report_dir/*"

    cd $TEST_DIR
    python -m pytest \
            --verbose -x -vv \
            --cov-config=$TEST_DIR/.coveragerc \
            --cov=$TEST_DIR/../ \
            $TEST_DIR 


    echo "Code coverage report conversion in JUnit format.."
    coverage xml -o "$coverage_report_dir/coverage_report.xml"
    coverage html -d "$coverage_report_dir/coverage_report_html"


    rm .coverage

    popd

fi