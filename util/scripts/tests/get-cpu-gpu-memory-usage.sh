#!/bin/bash
set -eou pipefail

usageExit() {
    echo "usage: $0 testPattern"
    echo "testPattern can include *'s so you can get metrics on multiple versions (CPU and GPU for example)"
    echo "pwd should be in the build directory, so we can run bin/tests"
    exit 1
}
[ -n "$1" ] || usageExit

[ -e bin/tests ] || {
    echo "bin/tests not found"
    usageExit
}

# For CPU memory usage:
# we use libmemusage.so which overrides *alloc and reports how much memory was used and what was the peak
# other alternatives for CPU memory (not implemented here):
# /usr/bin/time -v binary args...
#   this can report Maximim resident set size, this doesn't actually match used heap as things can be swapped out
# valgrind --tool=massif --massif-out-file=ms.out --depth=1 --time-unit=B --max-snapshots=10 binary args...
#   this will run the program through valgrind and monitor memory usage; valgrind is about 10x slower than libmemusage.so
#
# For GPU memory usage:
# in a separate monitoring process, we run nvidia-smi in a loop, sampling how much cuda memory our binary uses
# note: since we are sampling, this doesn't have perfect accuracy, but if the (large) memory is read from or to then
# it should take some time to do that, enough for us to sample; this may not properly be captured in the cases
# where we first allocate new memory and then release old memory without performing significant operations in between

libmemusage=/lib/x86_64-linux-gnu/libmemusage.so
[ -f "$libmemusage" ] || {
    echo "$libmemusage not found"
    exit 1
}

#make binary name unique to grep for
testbin=./tests-$$-bin
ln -s bin/tests $testbin
monitorGpuUsage() {
    while [ -f mongpu ]
    do
        nvidia-smi|grep "$testbin"||:
        sleep .01
    done|perl -pe 's/.* (\d+)MiB.*/$1/'|sort -n|tail -1 > mongpu.mib
    [ -n "$(cat mongpu.mib)" ] || echo 0 > mongpu.mib
    touch mongpudone
}
rm -f mongpudone
touch mongpu
monitorGpuUsage &

extractHeapAndInfo() {
    perl -ne '
    if (/^Open3dTestMemoryLimits /) {
        if (/^Open3dTestMemoryLimits test_name (.*) cpu_mb (\d+) gpu_mb (\d+) skip (\w+) device/) {
            if ($cm) {
                if ($cm != $2 or $gm != $3) {
                    die("Multiple Open3dTestMemoryLimits not matching $cm != $2 or $gm != $3");
                }
            }
            $cm=$2;
            $gm=$3;
            $skip+=($4 eq "true")?1:0;
        } else {
            die("Failed to parse Open3dTestMemoryLimits line: $_");
        }
    }
    if (/ heap total: \d+, heap peak: (\d+),/) {
        print "$1\n";
        if ($cm) { print "$cm $gm $skip\n"; }
    }
    '
}
#grab output from stderr and extract heap peak from it and Open3dTestMemoryLimits if present; still provide stdout on stderr
meminfo=$(OPEN3D_TEST_REPORT_MEMORY_LIMITS=1 LD_PRELOAD=$libmemusage $testbin --gtest_filter="$1" 3>&1 1>&2 2>&3 |extractHeapAndInfo)
#note: set -e -u pipefail will cause us to exit if bin/tests return failure
echo meminfo = $meminfo >&2
rm mongpu
rm $testbin

# it usually takes 2-4 sleep .01 for mongpu to finish
[ -f mongpudone ] || sleep .01
[ -f mongpudone ] || sleep .01
[ -f mongpudone ] || sleep .01
[ -f mongpudone ] || sleep .01
while [ ! -f mongpudone ];do echo "waiting for mongpu" >&2;sleep .01;done
gpumem=$(cat mongpu.mib)
rm mongpu.mib mongpudone
set -- $meminfo
cpumem=$1
((cpumib=cpumem/1048576+1))
out="cpumib $cpumib gpumib $gpumem"
[ "$#" -ge 2 ] && out="$out cpulimit $2 gpulimit $3 skip $4"
echo "$out"
