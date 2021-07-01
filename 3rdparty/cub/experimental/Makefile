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
# make <target> [sm=<XXX,...>] [cdp=<0|1>] [force32=<0|1>] [abi=<0|1>] [open64=<0|1>] [verbose=<0|1>] [keep=<0|1>] [quicktest=<0|1>]
#
#-------------------------------------------------------------------------------
 
include ../common.mk 

#-------------------------------------------------------------------------------
# Commandline Options
#-------------------------------------------------------------------------------

# [mkl=<0|1>] compile against Intel MKL
ifeq ($(mkl), 1)
	DEFINES 	+= -DCUB_MKL

ifeq (WIN_NT, $(findstring WIN_NT, $(OSUPPER)))
	LIBS 		+=	mkl_intel_lp64.lib mkl_intel_thread.lib  mkl_core.lib libiomp5md.lib
	NVCCFLAGS 	+= -Xcompiler /openmp
else
	LIBS		+= -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm
	NVCCFLAGS 	+= -Xcompiler -fopenmp
	
endif	

endif


#-------------------------------------------------------------------------------
# Compiler and compilation platform
#-------------------------------------------------------------------------------

# Includes
INC += -I$(CUB_DIR) -I$(CUB_DIR)test 

# detect OS
OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])

#-------------------------------------------------------------------------------
# Dependency Lists
#-------------------------------------------------------------------------------

exp_rwildcard=$(foreach d,$(wildcard $1*),$(call rwildcard,$d/,$2) $(filter $(subst *,%,$2),$d))

EXP_DEPS = 	$(call rwildcard, ./,*.cuh) \
			$(call rwildcard, ./,*.h)

DEPS =				$(CUB_DEPS) \
					$(EXP_DEPS) \
					$(CUB_DIR)test/Makefile \
					$(CUB_DIR)test/test_util.h \
					$(CUB_DIR)test/mersenne.h \

		

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
# make histogram_compare
#-------------------------------------------------------------------------------

histogram_compare: bin/histogram_compare_$(BIN_SUFFIX)

bin/histogram_compare_$(BIN_SUFFIX) : histogram_compare.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/histogram_compare_$(BIN_SUFFIX) histogram_compare.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3
	


#-------------------------------------------------------------------------------
# make spmv_compare
#-------------------------------------------------------------------------------

spmv_compare: bin/spmv_compare_$(BIN_SUFFIX)

bin/spmv_compare_$(BIN_SUFFIX) : spmv_compare.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/spmv_compare_$(BIN_SUFFIX) spmv_compare.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -lcusparse $(MKL_LIBS) -O3
	

