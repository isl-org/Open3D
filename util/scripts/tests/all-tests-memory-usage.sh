#!/bin/bash
set -eou pipefail

cpumib_mincheck=100
gpumib_mincheck=100
mib_tolerance=10

#we only trying to get the list of tests, for speed skip as many tests as we can
TEST_MAX_CPU_MEMORY_MB=1 TEST_MAX_GPU_MEMORY_MB=1 bin/tests "$@" | tee runtests.out
cat runtests.out|grep '\[ RUN '|
while read j0 j1 j2 name
do
    echo $name|perl -pe 's:(.*/)\d+:$1*:'
done | uniq > testlist.txt

rm -f testlimits.txt
cat testlist.txt|
while read test
do
    OPEN3D_TEST_REPORT_MEMORY_LIMITS=1 $(dirname $0)/get-cpu-gpu-memory-usage.sh "$test" > mibinfo.txt
    set -- $(cat mibinfo.txt)
    [ "$1" = "cpumib" -a "$3" = "gpumib" ] || {
        echo "Failed to collect memory usage information:"
        cat mibinfo.txt
        exit 1
    }
    cpumib=$2
    gpumib=$4
    testname=$(echo "$test"|perl -pe 's:^.*/([^/]*\.[^/]*)/.*$:$1:'|perl -pe 's:.*\.::')
    op="ADD:"
    needlimit=true
    havelimit=false
    missing_tests=false
    if [ "$#" -ge 5 ]
    then
        [ "$5" = "cpulimit" -a "$7" = "gpulimit" -a "$9" = "skip" ] || {
            echo "Failed to collect memory usage information:"
            cat mibinfo.txt
            exit 1
        }
        havelimit=true
        cpulimit=$6
        gpulimit=$8
        skip=${10}
        cpufar=true
        ((cpumib-mib_tolerance<cpulimit && cpulimit<cpumib+mib_tolerance)) && cpufar=false
        gpufar=true
        ((gpumib-mib_tolerance<gpulimit && gpulimit<gpumib+mib_tolerance)) && gpufar=false
        [ "$skip" = 0 ] || missing_tests=true
        if $missing_tests
        then
            echo "$test has not fully executed, skipped $skip types"
            op="UNKNOWN:"
        elif $cpufar||$gpufar
        then
            echo "$test has messed up parameters $cpulimit vs $cpumib or $gpulimit vs $gpumib"
            op="REPLACE:"
        else
            needlimit=false
            op="NEAR:"
        fi
    fi
    codeline="if (OverMemoryLimit(\"$testname\",$cpumib)) return;"
    [ $gpumib -gt 0 ] && codeline="if (OverMemoryLimit(\"$testname\",$cpumib,$gpumib,device)) return;"
    [ $gpumib -lt $gpumib_mincheck -a $cpumib -lt $cpumib_mincheck -a $havelimit = false ] && continue
    if $missing_tests
    then
        echo "MISSING DATA ($skip skipped): $test $op $codeline" >> testlimits.txt
        echo "MISSING DATA ($skip skipped): $test $op $codeline"
    elif $needlimit
    then
        echo "$test $op $codeline" >> testlimits.txt
        echo "$test $op $codeline"
    else
        echo "SKIP UPDATING: $test $op $codeline" >> testlimits.txt
        echo "SKIP UPDATING: $test $op $codeline"
    fi
done

echo
echo ======================================================================
if grep -v 'SKIP UPDATING:' < testlimits.txt
then
    rc=1
else
    rc=0
    echo "All tests marked correctly!"
fi
echo
echo "Full list in testlimits.txt"

exit $rc
