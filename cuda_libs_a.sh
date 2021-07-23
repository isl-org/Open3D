#!/bin/bash

echo "libcublasLt_static.a        " && nm -C /usr/local/cuda/lib64/libcublasLt_static.a          | grep dbdsqr_gpu
echo "libcublas_static.a          " && nm -C /usr/local/cuda/lib64/libcublas_static.a            | grep dbdsqr_gpu
echo "libcudadevrt.a              " && nm -C /usr/local/cuda/lib64/libcudadevrt.a                | grep dbdsqr_gpu
echo "libcudart_static.a          " && nm -C /usr/local/cuda/lib64/libcudart_static.a            | grep dbdsqr_gpu
echo "libcufft_static.a           " && nm -C /usr/local/cuda/lib64/libcufft_static.a             | grep dbdsqr_gpu
echo "libcufft_static_nocallback.a" && nm -C /usr/local/cuda/lib64/libcufft_static_nocallback.a  | grep dbdsqr_gpu
echo "libcufftw_static.a          " && nm -C /usr/local/cuda/lib64/libcufftw_static.a            | grep dbdsqr_gpu
echo "libculibos.a                " && nm -C /usr/local/cuda/lib64/libculibos.a                  | grep dbdsqr_gpu
echo "libcupti_static.a           " && nm -C /usr/local/cuda/lib64/libcupti_static.a             | grep dbdsqr_gpu
echo "libcurand_static.a          " && nm -C /usr/local/cuda/lib64/libcurand_static.a            | grep dbdsqr_gpu
echo "libcusolver_static.a        " && nm -C /usr/local/cuda/lib64/libcusolver_static.a          | grep dbdsqr_gpu
echo "libcusparse_static.a        " && nm -C /usr/local/cuda/lib64/libcusparse_static.a          | grep dbdsqr_gpu
echo "liblapack_static.a          " && nm -C /usr/local/cuda/lib64/liblapack_static.a            | grep dbdsqr_gpu
echo "libmetis_static.a           " && nm -C /usr/local/cuda/lib64/libmetis_static.a             | grep dbdsqr_gpu
echo "libnppc_static.a            " && nm -C /usr/local/cuda/lib64/libnppc_static.a              | grep dbdsqr_gpu
echo "libnppial_static.a          " && nm -C /usr/local/cuda/lib64/libnppial_static.a            | grep dbdsqr_gpu
echo "libnppicc_static.a          " && nm -C /usr/local/cuda/lib64/libnppicc_static.a            | grep dbdsqr_gpu
echo "libnppidei_static.a         " && nm -C /usr/local/cuda/lib64/libnppidei_static.a           | grep dbdsqr_gpu
echo "libnppif_static.a           " && nm -C /usr/local/cuda/lib64/libnppif_static.a             | grep dbdsqr_gpu
echo "libnppig_static.a           " && nm -C /usr/local/cuda/lib64/libnppig_static.a             | grep dbdsqr_gpu
echo "libnppim_static.a           " && nm -C /usr/local/cuda/lib64/libnppim_static.a             | grep dbdsqr_gpu
echo "libnppist_static.a          " && nm -C /usr/local/cuda/lib64/libnppist_static.a            | grep dbdsqr_gpu
echo "libnppisu_static.a          " && nm -C /usr/local/cuda/lib64/libnppisu_static.a            | grep dbdsqr_gpu
echo "libnppitc_static.a          " && nm -C /usr/local/cuda/lib64/libnppitc_static.a            | grep dbdsqr_gpu
echo "libnpps_static.a            " && nm -C /usr/local/cuda/lib64/libnpps_static.a              | grep dbdsqr_gpu
echo "libnvjpeg_static.a          " && nm -C /usr/local/cuda/lib64/libnvjpeg_static.a            | grep dbdsqr_gpu
echo "libnvperf_host_static.a     " && nm -C /usr/local/cuda/lib64/libnvperf_host_static.a       | grep dbdsqr_gpu
