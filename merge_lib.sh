#!/bin/bash

pushd cuda_lib
    ar x lib/libcublas_static.a
    ar x lib/libcublasLt_static.a
    ar x lib/liblapack_static.a
    # ar x lib/libculibos.a

    ar -qc lib/libcuda_merged.a *.o
    echo "lib/libcuda_merged.a  : " && nm -C lib/libcuda_merged.a   | grep cublasSgemmEx
    echo "lib/libcublas_static.a: " && nm -C lib/libcublas_static.a | grep cublasSgemmEx
popd
