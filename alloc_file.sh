#!/usr/bin/env bash

for i in {1..10000}
do
   echo "Allocating file #$i"
   truncate -s 1M "$i.tmp"
done
