#!/bin/bash

pushd cuda_lib
    rm -rf *.o

    rm -rf libcublas_static   && mkdir libcublas_static   && ar x lib/libcublas_static.a   && mv *.o libcublas_static   && rm -rf *.o && ls -al libcublas_static/*.o   | wc -l
    rm -rf libcublasLt_static && mkdir libcublasLt_static && ar x lib/libcublasLt_static.a && mv *.o libcublasLt_static && rm -rf *.o && ls -al libcublasLt_static/*.o | wc -l
    rm -rf liblapack_static   && mkdir liblapack_static   && ar x lib/liblapack_static.a   && mv *.o liblapack_static   && rm -rf *.o && ls -al liblapack_static/*.o   | wc -l
    rm -rf libculibos         && mkdir libculibos         && ar x lib/libculibos.a         && mv *.o libculibos         && rm -rf *.o && ls -al libculibos/*.o         | wc -l

    cp libcublas_static/*.o   .
    cp libcublasLt_static/*.o .
    cp liblapack_static/*.o   .
    cp libculibos/*.o         .
    ls -al *.o | wc -l
    rm -rf *.o

    ar -qc lib/libcuda_merged.a libcublas_static/*.o libcublasLt_static/*.o liblapack_static/*.o libculibos/*.o
    echo "lib/libcuda_merged.a  : " && nm -C lib/libcuda_merged.a   | grep cublasSgemmEx
    echo "lib/libcublas_static.a: " && nm -C lib/libcublas_static.a | grep cublasSgemmEx
popd
