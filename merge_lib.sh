#!/bin/bash

pushd cuda_lib
    ls
    rm -rf *.o

    rm -rf libcublas_static   && mkdir libcublas_static   && ar x lib/libcublas_static.a   && mv *.o libcublas_static   && rm -rf *.o
    rm -rf libcublasLt_static && mkdir libcublasLt_static && ar x lib/libcublasLt_static.a && mv *.o libcublasLt_static && rm -rf *.o
    rm -rf liblapack_static   && mkdir liblapack_static   && ar x lib/liblapack_static.a   && mv *.o liblapack_static   && rm -rf *.o
    rm -rf libculibos         && mkdir libculibos         && ar x lib/libculibos.a         && mv *.o libculibos         && rm -rf *.o
popd
