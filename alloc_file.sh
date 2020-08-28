#!/usr/bin/env bash

for i in {1..100000}
do
   echo "Allocating file #$i"
   truncate -s 10M "$i.tmp"
done
