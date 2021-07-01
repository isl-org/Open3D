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
# Build script for project
#-------------------------------------------------------------------------------

NVCC = "$(shell which nvcc)"
NVCC_VERSION = $(strip $(shell nvcc --version | grep release | sed 's/.*release //' |  sed 's/,.*//'))

# detect OS
OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])

#-------------------------------------------------------------------------------
# Libs
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Includes
#-------------------------------------------------------------------------------

INC = -I. -I.. -I../test

#-------------------------------------------------------------------------------
# Libs
#-------------------------------------------------------------------------------

LIBS += -lcudart 

#-------------------------------------------------------------------------------
# Defines
#-------------------------------------------------------------------------------

DEFINES = 

#-------------------------------------------------------------------------------
# SM Arch
#-------------------------------------------------------------------------------

ifdef sm
	SM_ARCH = $(sm)
else 
    SM_ARCH = 200
endif

# Only one arch per tuning binary
ifeq (350, $(findstring 350, $(SM_ARCH)))
    SM_TARGETS = -arch=sm_35
    SM_ARCH = 350
endif
ifeq (300, $(findstring 300, $(SM_ARCH)))
    SM_TARGETS = -arch=sm_30
    SM_ARCH = 300
endif
ifeq (200, $(findstring 200, $(SM_ARCH)))
    SM_TARGETS = -arch=sm_20
    SM_ARCH = 200
endif
ifeq (130, $(findstring 130, $(SM_ARCH)))
    SM_TARGETS = -arch=sm_13
    SM_ARCH = 130
endif
ifeq (110, $(findstring 110, $(SM_ARCH)))
    SM_TARGETS = -arch=sm_11 
    SM_ARCH = 110
endif
ifeq (100, $(findstring 100, $(SM_ARCH)))
    SM_TARGETS = -arch=sm_10 
    SM_ARCH = 100
endif


#-------------------------------------------------------------------------------
# Compiler Flags
#-------------------------------------------------------------------------------

NVCCFLAGS = -Xptxas -v -Xcudafe -\#

# Help the compiler/linker work with huge numbers of kernels on Windows
ifeq (WIN_NT, $(findstring WIN_NT, $(OSUPPER)))
	NVCCFLAGS += -Xcompiler /bigobj -Xcompiler /Zm500
endif

# 32/64-bit (32-bit device pointers by default) 
ifeq ($(force32), 1)
	CPU_ARCH = -m32
	CPU_ARCH_SUFFIX = i386
else
	CPU_ARCH = -m64
	CPU_ARCH_SUFFIX = x86_64
endif

# CUDA ABI enable/disable (enabled by default) 
ifneq ($(abi), 0)
	ABI_SUFFIX = abi
else 
	NVCCFLAGS += -Xptxas -abi=no
	ABI_SUFFIX = noabi
endif

# NVVM/Open64 middle-end compiler (nvvm by default)
ifeq ($(open64), 1)
	NVCCFLAGS += -open64
	PTX_SUFFIX = open64
else 
	PTX_SUFFIX = nvvm
endif

# Verbose toolchain output from nvcc
ifeq ($(verbose), 1)
	NVCCFLAGS += -v
endif

# Keep intermediate compilation artifacts
ifeq ($(keep), 1)
	NVCCFLAGS += -keep
endif

# Data type size to compile a schmoo binary for
ifdef tunesize
    TUNE_SIZE = $(tunesize)
else 
	TUNE_SIZE = 4
endif


SUFFIX = $(TUNE_SIZE)B_sm$(SM_ARCH)_$(PTX_SUFFIX)_$(NVCC_VERSION)_$(ABI_SUFFIX)_$(CPU_ARCH_SUFFIX)

#-------------------------------------------------------------------------------
# Dependency Lists
#-------------------------------------------------------------------------------

rwildcard=$(foreach d,$(wildcard $1*),$(call rwildcard,$d/,$2) $(filter $(subst *,%,$2),$d))

DEPS =	 ./Makefile \
		../test/test_util.h \
		$(call rwildcard,../cub/,*.cuh)


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
# make tune_device_reduce
#-------------------------------------------------------------------------------

tune_device_reduce: bin/tune_device_reduce_$(SUFFIX)

bin/tune_device_reduce_$(SUFFIX) : tune_device_reduce.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/tune_device_reduce_$(SUFFIX) tune_device_reduce.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3 -DTUNE_ARCH=$(SM_ARCH) -DTUNE_SIZE=$(TUNE_SIZE)

