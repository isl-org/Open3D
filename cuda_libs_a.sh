#!/bin/bash

echo "libcublasLt_static.a        " && nm -C /usr/local/cuda/lib64/libcublasLt_static.a          | grep cublasSgemmEx
echo "libcublas_static.a          " && nm -C /usr/local/cuda/lib64/libcublas_static.a            | grep cublasSgemmEx
echo "libcudadevrt.a              " && nm -C /usr/local/cuda/lib64/libcudadevrt.a                | grep cublasSgemmEx
echo "libcudart_static.a          " && nm -C /usr/local/cuda/lib64/libcudart_static.a            | grep cublasSgemmEx
echo "libcufft_static.a           " && nm -C /usr/local/cuda/lib64/libcufft_static.a             | grep cublasSgemmEx
echo "libcufft_static_nocallback.a" && nm -C /usr/local/cuda/lib64/libcufft_static_nocallback.a  | grep cublasSgemmEx
echo "libcufftw_static.a          " && nm -C /usr/local/cuda/lib64/libcufftw_static.a            | grep cublasSgemmEx
echo "libculibos.a                " && nm -C /usr/local/cuda/lib64/libculibos.a                  | grep cublasSgemmEx
echo "libcupti_static.a           " && nm -C /usr/local/cuda/lib64/libcupti_static.a             | grep cublasSgemmEx
echo "libcurand_static.a          " && nm -C /usr/local/cuda/lib64/libcurand_static.a            | grep cublasSgemmEx
echo "libcusolver_static.a        " && nm -C /usr/local/cuda/lib64/libcusolver_static.a          | grep cublasSgemmEx
echo "libcusparse_static.a        " && nm -C /usr/local/cuda/lib64/libcusparse_static.a          | grep cublasSgemmEx
echo "liblapack_static.a          " && nm -C /usr/local/cuda/lib64/liblapack_static.a            | grep cublasSgemmEx
echo "libmetis_static.a           " && nm -C /usr/local/cuda/lib64/libmetis_static.a             | grep cublasSgemmEx
echo "libnppc_static.a            " && nm -C /usr/local/cuda/lib64/libnppc_static.a              | grep cublasSgemmEx
echo "libnppial_static.a          " && nm -C /usr/local/cuda/lib64/libnppial_static.a            | grep cublasSgemmEx
echo "libnppicc_static.a          " && nm -C /usr/local/cuda/lib64/libnppicc_static.a            | grep cublasSgemmEx
echo "libnppidei_static.a         " && nm -C /usr/local/cuda/lib64/libnppidei_static.a           | grep cublasSgemmEx
echo "libnppif_static.a           " && nm -C /usr/local/cuda/lib64/libnppif_static.a             | grep cublasSgemmEx
echo "libnppig_static.a           " && nm -C /usr/local/cuda/lib64/libnppig_static.a             | grep cublasSgemmEx
echo "libnppim_static.a           " && nm -C /usr/local/cuda/lib64/libnppim_static.a             | grep cublasSgemmEx
echo "libnppist_static.a          " && nm -C /usr/local/cuda/lib64/libnppist_static.a            | grep cublasSgemmEx
echo "libnppisu_static.a          " && nm -C /usr/local/cuda/lib64/libnppisu_static.a            | grep cublasSgemmEx
echo "libnppitc_static.a          " && nm -C /usr/local/cuda/lib64/libnppitc_static.a            | grep cublasSgemmEx
echo "libnpps_static.a            " && nm -C /usr/local/cuda/lib64/libnpps_static.a              | grep cublasSgemmEx
echo "libnvjpeg_static.a          " && nm -C /usr/local/cuda/lib64/libnvjpeg_static.a            | grep cublasSgemmEx
echo "libnvperf_host_static.a     " && nm -C /usr/local/cuda/lib64/libnvperf_host_static.a       | grep cublasSgemmEx
