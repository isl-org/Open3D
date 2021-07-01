#/******************************************************************************
# * Copyright (c) 2011, Duane Merrill.  All rights reserved.
# * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
# * 
# * Redistribution and use in source and binary forms, with or without
# * modification, are permitted provided that the following conditions are met:
# *	 * Redistributions of source code must retain the above copyright
# *	   notice, this list of conditions and the following disclaimer.
# *	 * Redistributions in binary form must reproduce the above copyright
# *	   notice, this list of conditions and the following disclaimer in the
# *	   documentation and/or other materials provided with the distribution.
# *	 * Neither the name of the NVIDIA CORPORATION nor the
# *	   names of its contributors may be used to endorse or promote products
# *	   derived from this software without specific prior written permission.
# * 
# * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
# * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# *
#******************************************************************************/


#-------------------------------------------------------------------------------
#
# Makefile usage
#
# make <target> [sm=<XXX,...>] [cdp=<0|1>] [force32=<0|1>] [abi=<0|1>] [open64=<0|1>] [verbose=<0|1>] [keep=<0|1>] [quicktest=<0|1>] [quickertest=<0|1>]
#
#-------------------------------------------------------------------------------

include ../common.mk 
 
#-------------------------------------------------------------------------------
# Commandline Options
#-------------------------------------------------------------------------------

# Testing mode option (quick/thorough)
ifeq ($(quickertest), 1)
	NVCCFLAGS += -DQUICKER_TEST
	TEST_SUFFIX = quicker
else ifeq ($(quicktest), 1)
	NVCCFLAGS += -DQUICK_TEST
	TEST_SUFFIX = quick
else 
	TEST_SUFFIX = thorough
	NPPI = 
endif


# CUDA memcheck (enabled by default) 
ifeq ($(memcheck), 0)
	MEMCHECK = 
else 
	MEMCHECK = cuda-memcheck
endif


#-------------------------------------------------------------------------------
# Compiler and compilation platform
#-------------------------------------------------------------------------------

# Includes
INC += -I$(CUB_DIR) -I$(CUB_DIR)test 

# Suffix to append to each binary
SUFFIX = $(BIN_SUFFIX)_$(TEST_SUFFIX)

# Define test arch
DEFINES += -DTEST_ARCH=$(TEST_ARCH)


#-------------------------------------------------------------------------------
# Dependency Lists
#-------------------------------------------------------------------------------

rwildcard=$(foreach d,$(wildcard $1*),$(call rwildcard,$d/,$2) $(filter $(subst *,%,$2),$d))

DEPS =				$(CUB_DEPS) \
					$(CUB_DIR)test/Makefile \
					$(CUB_DIR)test/test_util.h \
					$(CUB_DIR)test/mersenne.h \

BLOCK_REDUCE = 		test_block_reduce_raking \
	 				test_block_reduce_warp_reductions		


BLOCK_SCAN = 		test_block_scan_raking \
	 				test_block_scan_raking_memoize \
	 				test_block_scan_warp_scans		


BLOCK_RADIX_SORT = 	test_block_radix_sort_keys \
	 				test_block_radix_sort_pairs	

DEVICE_RADIX_SORT = 	test_device_radix_sort \
	 					test_device_radix_sort_segmented	
		
ALL = 				link \
	 				test_iterator \
	 				test_allocator \
	 				test_warp_scan \
	 				test_warp_reduce \
	 				$(BLOCK_REDUCE) \
	 				$(BLOCK_SCAN) \
	 				$(BLOCK_RADIX_SORT) \
	 				test_block_load_store \
	 				test_block_histogram \
				 	test_device_reduce \
			 		test_device_histogram \
			 		test_device_scan \
	 				$(DEVICE_RADIX_SORT) \
					test_device_reduce_by_key\
					test_device_run_length_encode\
		 			test_device_select_unique \
					test_device_select_if 
		
#	 	test_grid_barrier \		fails on sm110
#	 	test_device_seg_reduce
		


#-------------------------------------------------------------------------------
# make default
#-------------------------------------------------------------------------------

default:


#-------------------------------------------------------------------------------
# make clean
#-------------------------------------------------------------------------------

clean :
	rm -f bin/*$(CPU_ARCH_SUFFIX)* 
	rm -f *.i* *.cubin *.cu.c *.cudafe* *.fatbin.c *.ptx *.hash *.cu.cpp *.o


#-------------------------------------------------------------------------------
# make all
#-------------------------------------------------------------------------------

all : $(ALL)


#-------------------------------------------------------------------------------
# make run
#-------------------------------------------------------------------------------

run : 
	for i in $(ALL); do $(MEMCHECK) ./bin/$${i}_$(SUFFIX) --device=$(device) || exit 1; done

run_block_reduce : 
	for i in $(BLOCK_REDUCE); do $(MEMCHECK) ./bin/$${i}_$(SUFFIX) --device=$(device) || exit 1; done

run_block_scan : 
	for i in $(BLOCK_SCAN); do $(MEMCHECK) ./bin/$${i}_$(SUFFIX) --device=$(device) || exit 1; done

run_block_radix_sort : 
	for i in $(BLOCK_RADIX_SORT); do $(MEMCHECK) ./bin/$${i}_$(SUFFIX) --device=$(device) || exit 1; done

run_device_radix_sort : 
	for i in $(DEVICE_RADIX_SORT); do $(MEMCHECK) ./bin/$${i}_$(SUFFIX) --device=$(device) || exit 1; done


#-------------------------------------------------------------------------------
# make link
#-------------------------------------------------------------------------------

link : bin/link_$(SUFFIX)

bin/link_$(SUFFIX) : link_a.cu link_b.cu link_main.cpp $(DEPS)
	mkdir -p bin
	$(NVCC) $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(DEFINES) $(SM_TARGETS) link_a.cu -c -o bin/link_a.obj
	$(NVCC) $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(DEFINES) $(SM_TARGETS) link_b.cu -c -o bin/link_b.obj
	$(NVCC) $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(DEFINES) $(SM_TARGETS) link_main.cpp bin/link_a.obj bin/link_b.obj -o bin/link_$(SUFFIX)


#-------------------------------------------------------------------------------
# make test_iterator 
#-------------------------------------------------------------------------------

test_iterator: bin/test_iterator_$(SUFFIX)

bin/test_iterator_$(SUFFIX) : test_iterator.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/test_iterator_$(SUFFIX) test_iterator.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3


#-------------------------------------------------------------------------------
# make test_allocator 
#-------------------------------------------------------------------------------

test_allocator: bin/test_allocator_$(SUFFIX)

bin/test_allocator_$(SUFFIX) : test_allocator.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/test_allocator_$(SUFFIX) test_allocator.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3
	
	
#-------------------------------------------------------------------------------
# make test_grid_barrier 
#-------------------------------------------------------------------------------

test_grid_barrier: bin/test_grid_barrier_$(SUFFIX)

bin/test_grid_barrier_$(SUFFIX) : test_grid_barrier.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/test_grid_barrier_$(SUFFIX) test_grid_barrier.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3  
	

#-------------------------------------------------------------------------------
# make test_warp_scan 
#-------------------------------------------------------------------------------

test_warp_scan: bin/test_warp_scan_$(SUFFIX)

bin/test_warp_scan_$(SUFFIX) : test_warp_scan.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/test_warp_scan_$(SUFFIX) test_warp_scan.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3  


#-------------------------------------------------------------------------------
# make test_warp_reduce 
#-------------------------------------------------------------------------------

test_warp_reduce: bin/test_warp_reduce_$(SUFFIX)

bin/test_warp_reduce_$(SUFFIX) : test_warp_reduce.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/test_warp_reduce_$(SUFFIX) test_warp_reduce.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3  


#-------------------------------------------------------------------------------
# make test_block_reduce_raking
#-------------------------------------------------------------------------------

test_block_reduce_raking: bin/test_block_reduce_raking_$(SUFFIX)

bin/test_block_reduce_raking_$(SUFFIX) : test_block_reduce.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) -DTEST_RAKING $(SM_TARGETS) -o bin/test_block_reduce_raking_$(SUFFIX) test_block_reduce.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3  


#-------------------------------------------------------------------------------
# make test_block_reduce_warp_reductions 
#-------------------------------------------------------------------------------

test_block_reduce_warp_reductions: bin/test_block_reduce_warp_reductions_$(SUFFIX)

bin/test_block_reduce_warp_reductions_$(SUFFIX) : test_block_reduce.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) -DTEST_WARP_REDUCTIONS $(SM_TARGETS) -o bin/test_block_reduce_warp_reductions_$(SUFFIX) test_block_reduce.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3  


#-------------------------------------------------------------------------------
# make test_block_reduce 
#-------------------------------------------------------------------------------

test_block_reduce: $(BLOCK_REDUCE)


#-------------------------------------------------------------------------------
# make test_block_scan_raking
#-------------------------------------------------------------------------------

test_block_scan_raking: bin/test_block_scan_raking_$(SUFFIX)

bin/test_block_scan_raking_$(SUFFIX) : test_block_scan.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) -DTEST_RAKING $(SM_TARGETS) -o bin/test_block_scan_raking_$(SUFFIX) test_block_scan.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3  


#-------------------------------------------------------------------------------
# make test_block_scan_raking_memoize
#-------------------------------------------------------------------------------

test_block_scan_raking_memoize: bin/test_block_scan_raking_memoize_$(SUFFIX)

bin/test_block_scan_raking_memoize_$(SUFFIX) : test_block_scan.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) -DTEST_RAKING_MEMOIZE $(SM_TARGETS) -o bin/test_block_scan_raking_memoize_$(SUFFIX) test_block_scan.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3  


#-------------------------------------------------------------------------------
# make test_block_scan_warp_scans
#-------------------------------------------------------------------------------

test_block_scan_warp_scans: bin/test_block_scan_warp_scans_$(SUFFIX)

bin/test_block_scan_warp_scans_$(SUFFIX) : test_block_scan.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) -DTEST_WARP_SCANS $(SM_TARGETS) -o bin/test_block_scan_warp_scans_$(SUFFIX) test_block_scan.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3  


#-------------------------------------------------------------------------------
# make test_block_scan 
#-------------------------------------------------------------------------------

test_block_scan: $(BLOCK_SCAN)


#-------------------------------------------------------------------------------
# make test_block_load_store 
#-------------------------------------------------------------------------------

test_block_load_store: bin/test_block_load_store_$(SUFFIX)

bin/test_block_load_store_$(SUFFIX) : test_block_load_store.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/test_block_load_store_$(SUFFIX) test_block_load_store.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3
	
	
#-------------------------------------------------------------------------------
# make test_block_radix_sort_keys 
#-------------------------------------------------------------------------------

test_block_radix_sort_keys: bin/test_block_radix_sort_keys_$(SUFFIX)

bin/test_block_radix_sort_keys_$(SUFFIX) : test_block_radix_sort.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) -DTEST_KEYS_ONLY $(SM_TARGETS) -o bin/test_block_radix_sort_keys_$(SUFFIX) test_block_radix_sort.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3

#-------------------------------------------------------------------------------
# make test_block_radix_sort_pairs 
#-------------------------------------------------------------------------------

test_block_radix_sort_pairs: bin/test_block_radix_sort_pairs_$(SUFFIX)

bin/test_block_radix_sort_pairs_$(SUFFIX) : test_block_radix_sort.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/test_block_radix_sort_pairs_$(SUFFIX) test_block_radix_sort.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3


#-------------------------------------------------------------------------------
# make test_block_radix_sort
#-------------------------------------------------------------------------------

test_block_radix_sort : $(BLOCK_RADIX_SORT)


#-------------------------------------------------------------------------------
# make test_block_histogram 
#-------------------------------------------------------------------------------

test_block_histogram: bin/test_block_histogram_$(SUFFIX)

bin/test_block_histogram_$(SUFFIX) : test_block_histogram.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/test_block_histogram_$(SUFFIX) test_block_histogram.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3


#-------------------------------------------------------------------------------
# make test_device_reduce
#-------------------------------------------------------------------------------

test_device_reduce: bin/test_device_reduce_$(SUFFIX)

bin/test_device_reduce_$(SUFFIX) : test_device_reduce.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/test_device_reduce_$(SUFFIX) test_device_reduce.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3


#-------------------------------------------------------------------------------
# make test_device_histogram
#-------------------------------------------------------------------------------

test_device_histogram: bin/test_device_histogram_$(SUFFIX)

bin/test_device_histogram_$(SUFFIX) : test_device_histogram.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/test_device_histogram_$(SUFFIX) test_device_histogram.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) $(NPPI) -O3


#-------------------------------------------------------------------------------
# make test_device_scan
#-------------------------------------------------------------------------------

test_device_scan: bin/test_device_scan_$(SUFFIX)

bin/test_device_scan_$(SUFFIX) : test_device_scan.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/test_device_scan_$(SUFFIX) test_device_scan.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3


#-------------------------------------------------------------------------------
# make test_device_radix_sort
#-------------------------------------------------------------------------------

test_device_radix_sort: bin/test_device_radix_sort_$(SUFFIX)

bin/test_device_radix_sort_$(SUFFIX) : test_device_radix_sort.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/test_device_radix_sort_$(SUFFIX) test_device_radix_sort.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3


#-------------------------------------------------------------------------------
# make test_device_radix_sort_segmented
#-------------------------------------------------------------------------------

test_device_radix_sort_segmented: bin/test_device_radix_sort_segmented_$(SUFFIX)

bin/test_device_radix_sort_segmented_$(SUFFIX) : test_device_radix_sort.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) -DSEGMENTED_SORT $(SM_TARGETS) -o bin/test_device_radix_sort_segmented_$(SUFFIX) test_device_radix_sort.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3


#-------------------------------------------------------------------------------
# make test_device_select_unique
#-------------------------------------------------------------------------------

test_device_select_unique: bin/test_device_select_unique_$(SUFFIX)

bin/test_device_select_unique_$(SUFFIX) : test_device_select_unique.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/test_device_select_unique_$(SUFFIX) test_device_select_unique.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3


#-------------------------------------------------------------------------------
# make test_device_select_if
#-------------------------------------------------------------------------------

test_device_select_if: bin/test_device_select_if_$(SUFFIX)

bin/test_device_select_if_$(SUFFIX) : test_device_select_if.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/test_device_select_if_$(SUFFIX) test_device_select_if.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3

#-------------------------------------------------------------------------------
# make test_device_reduce_by_key
#-------------------------------------------------------------------------------

test_device_reduce_by_key: bin/test_device_reduce_by_key_$(SUFFIX)

bin/test_device_reduce_by_key_$(SUFFIX) : test_device_reduce_by_key.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/test_device_reduce_by_key_$(SUFFIX) test_device_reduce_by_key.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3

#-------------------------------------------------------------------------------
# make test_device_run_length_encode
#-------------------------------------------------------------------------------

test_device_run_length_encode: bin/test_device_run_length_encode_$(SUFFIX)

bin/test_device_run_length_encode_$(SUFFIX) : test_device_run_length_encode.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/test_device_run_length_encode_$(SUFFIX) test_device_run_length_encode.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3




#-------------------------------------------------------------------------------
# make test_device_seg_reduce
#-------------------------------------------------------------------------------
#
#test_device_seg_reduce: bin/test_device_seg_reduce_$(SUFFIX)
#
#bin/test_device_seg_reduce_$(SUFFIX) : test_device_seg_reduce.cu $(DEPS)
#	mkdir -p bin
#	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/test_device_seg_reduce_$(SUFFIX) test_device_seg_reduce.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3


