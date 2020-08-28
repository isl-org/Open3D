#!/usr/bin/env bash

for i in {1..100000}
do
   echo "Allocating file #$i"
   head -c 1048576 </dev/urandom >$i.tmp
done
