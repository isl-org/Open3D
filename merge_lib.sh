#!/bin/bash

pushd cuda_lib
    rm -rf *.o

    rm -rf libcublas_static   && mkdir libcublas_static   && ar x lib/libcublas_static.a   && mv *.o libcublas_static   && rm -rf *.o && ls -alh libcublas_static   | wc -l
    rm -rf libcublasLt_static && mkdir libcublasLt_static && ar x lib/libcublasLt_static.a && mv *.o libcublasLt_static && rm -rf *.o && ls -alh libcublasLt_static | wc -l
    rm -rf liblapack_static   && mkdir liblapack_static   && ar x lib/liblapack_static.a   && mv *.o liblapack_static   && rm -rf *.o && ls -alh liblapack_static   | wc -l
    rm -rf libculibos         && mkdir libculibos         && ar x lib/libculibos.a         && mv *.o libculibos         && rm -rf *.o && ls -alh libculibos         | wc -l
popd
