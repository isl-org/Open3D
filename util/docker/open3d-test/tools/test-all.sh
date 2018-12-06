#!/bin/sh

echo "testing all images..."
echo

for ubuntu in 14.04 16.04 18.04; do
    ./build.sh $ubuntu base
    echo

    for python in py2 py3; do
        for deps in no_deps with_deps; do
            ./test.sh $ubuntu $python $deps
            echo
        done
    done
done

# display images in order to check image size
docker image ls
echo
