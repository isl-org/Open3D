#!/bin/bash

ITEM=cublasLtGetProperty

nm -D -C /usr/local/cuda/lib64/libaccinj64.so       | grep ${ITEM}
nm -D -C /usr/local/cuda/lib64/libcublasLt.so       | grep ${ITEM}
nm -D -C /usr/local/cuda/lib64/libcublas.so         | grep ${ITEM}
nm -D -C /usr/local/cuda/lib64/libcudart.so         | grep ${ITEM}
nm -D -C /usr/local/cuda/lib64/libcufft.so          | grep ${ITEM}
nm -D -C /usr/local/cuda/lib64/libcufftw.so         | grep ${ITEM}
nm -D -C /usr/local/cuda/lib64/libcuinj64.so        | grep ${ITEM}
nm -D -C /usr/local/cuda/lib64/libcupti.so          | grep ${ITEM}
nm -D -C /usr/local/cuda/lib64/libcurand.so         | grep ${ITEM}
nm -D -C /usr/local/cuda/lib64/libcusolverMg.so     | grep ${ITEM}
nm -D -C /usr/local/cuda/lib64/libcusolver.so       | grep ${ITEM}
nm -D -C /usr/local/cuda/lib64/libcusparse.so       | grep ${ITEM}
nm -D -C /usr/local/cuda/lib64/libnppc.so           | grep ${ITEM}
nm -D -C /usr/local/cuda/lib64/libnppial.so         | grep ${ITEM}
nm -D -C /usr/local/cuda/lib64/libnppicc.so         | grep ${ITEM}
nm -D -C /usr/local/cuda/lib64/libnppidei.so        | grep ${ITEM}
nm -D -C /usr/local/cuda/lib64/libnppif.so          | grep ${ITEM}
nm -D -C /usr/local/cuda/lib64/libnppig.so          | grep ${ITEM}
nm -D -C /usr/local/cuda/lib64/libnppim.so          | grep ${ITEM}
nm -D -C /usr/local/cuda/lib64/libnppist.so         | grep ${ITEM}
nm -D -C /usr/local/cuda/lib64/libnppisu.so         | grep ${ITEM}
nm -D -C /usr/local/cuda/lib64/libnppitc.so         | grep ${ITEM}
nm -D -C /usr/local/cuda/lib64/libnpps.so           | grep ${ITEM}
nm -D -C /usr/local/cuda/lib64/libnvblas.so         | grep ${ITEM}
nm -D -C /usr/local/cuda/lib64/libnvjpeg.so         | grep ${ITEM}
nm -D -C /usr/local/cuda/lib64/libnvperf_host.so    | grep ${ITEM}
nm -D -C /usr/local/cuda/lib64/libnvperf_target.so  | grep ${ITEM}
nm -D -C /usr/local/cuda/lib64/libnvrtc-builtins.so | grep ${ITEM}
nm -D -C /usr/local/cuda/lib64/libnvrtc.so          | grep ${ITEM}
nm -D -C /usr/local/cuda/lib64/libnvToolsExt.so     | grep ${ITEM}
nm -D -C /usr/local/cuda/lib64/libOpenCL.so         | grep ${ITEM}
