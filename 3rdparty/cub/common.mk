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
# Commandline Options
#-------------------------------------------------------------------------------

# [sm=<XXX,...>] Compute-capability to compile for, e.g., "sm=200,300,350" (SM20 by default).
  
COMMA = ,
ifdef sm
	SM_ARCH = $(subst $(COMMA),-,$(sm))
else 
    SM_ARCH = 200
endif

ifeq (700, $(findstring 700, $(SM_ARCH)))
    SM_TARGETS 	+= -gencode=arch=compute_70,code=\"sm_70,compute_70\" 
    SM_DEF 		+= -DSM700
    TEST_ARCH 	= 700
endif
ifeq (620, $(findstring 620, $(SM_ARCH)))
    SM_TARGETS 	+= -gencode=arch=compute_62,code=\"sm_62,compute_62\" 
    SM_DEF 		+= -DSM620
    TEST_ARCH 	= 620
endif
ifeq (610, $(findstring 610, $(SM_ARCH)))
    SM_TARGETS 	+= -gencode=arch=compute_61,code=\"sm_61,compute_61\" 
    SM_DEF 		+= -DSM610
    TEST_ARCH 	= 610
endif
ifeq (600, $(findstring 600, $(SM_ARCH)))
    SM_TARGETS 	+= -gencode=arch=compute_60,code=\"sm_60,compute_60\" 
    SM_DEF 		+= -DSM600
    TEST_ARCH 	= 600
endif
ifeq (520, $(findstring 520, $(SM_ARCH)))
    SM_TARGETS 	+= -gencode=arch=compute_52,code=\"sm_52,compute_52\" 
    SM_DEF 		+= -DSM520
    TEST_ARCH 	= 520
endif
ifeq (370, $(findstring 370, $(SM_ARCH)))
    SM_TARGETS 	+= -gencode=arch=compute_37,code=\"sm_37,compute_37\" 
    SM_DEF 		+= -DSM370
    TEST_ARCH 	= 370
endif
ifeq (350, $(findstring 350, $(SM_ARCH)))
    SM_TARGETS 	+= -gencode=arch=compute_35,code=\"sm_35,compute_35\" 
    SM_DEF 		+= -DSM350
    TEST_ARCH 	= 350
endif
ifeq (300, $(findstring 300, $(SM_ARCH)))
    SM_TARGETS 	+= -gencode=arch=compute_30,code=\"sm_30,compute_30\"
    SM_DEF 		+= -DSM300
    TEST_ARCH 	= 300
endif
ifeq (210, $(findstring 210, $(SM_ARCH)))
    SM_TARGETS 	+= -gencode=arch=compute_20,code=\"sm_21,compute_20\"
    SM_DEF 		+= -DSM210
    TEST_ARCH 	= 210
endif
ifeq (200, $(findstring 200, $(SM_ARCH)))
    SM_TARGETS 	+= -gencode=arch=compute_20,code=\"sm_20,compute_20\"
    SM_DEF 		+= -DSM200
    TEST_ARCH 	= 200
endif
ifeq (130, $(findstring 130, $(SM_ARCH)))
    SM_TARGETS 	+= -gencode=arch=compute_13,code=\"sm_13,compute_13\" 
    SM_DEF 		+= -DSM130
    TEST_ARCH 	= 130
endif
ifeq (120, $(findstring 120, $(SM_ARCH)))
    SM_TARGETS 	+= -gencode=arch=compute_12,code=\"sm_12,compute_12\" 
    SM_DEF 		+= -DSM120
    TEST_ARCH 	= 120
endif
ifeq (110, $(findstring 110, $(SM_ARCH)))
    SM_TARGETS 	+= -gencode=arch=compute_11,code=\"sm_11,compute_11\" 
    SM_DEF 		+= -DSM110
    TEST_ARCH 	= 110
endif
ifeq (100, $(findstring 100, $(SM_ARCH)))
    SM_TARGETS 	+= -gencode=arch=compute_10,code=\"sm_10,compute_10\" 
    SM_DEF 		+= -DSM100
    TEST_ARCH 	= 100
endif


# [cdp=<0|1>] CDP enable option (default: no)
ifeq ($(cdp), 1)
	DEFINES += -DCUB_CDP
	CDP_SUFFIX = cdp
    NVCCFLAGS += -rdc=true -lcudadevrt
else
	CDP_SUFFIX = nocdp
endif


# [force32=<0|1>] Device addressing mode option (64-bit device pointers by default) 
ifeq ($(force32), 1)
	CPU_ARCH = -m32
	CPU_ARCH_SUFFIX = i386
else
	CPU_ARCH = -m64
	CPU_ARCH_SUFFIX = x86_64
    NPPI = -lnppist
endif


# [abi=<0|1>] CUDA ABI option (enabled by default) 
ifneq ($(abi), 0)
	ABI_SUFFIX = abi
else 
	NVCCFLAGS += -Xptxas -abi=no
	ABI_SUFFIX = noabi
endif


# [open64=<0|1>] Middle-end compiler option (nvvm by default)
ifeq ($(open64), 1)
	NVCCFLAGS += -open64
	PTX_SUFFIX = open64
else 
	PTX_SUFFIX = nvvm
endif


# [verbose=<0|1>] Verbose toolchain output from nvcc option
ifeq ($(verbose), 1)
	NVCCFLAGS += -v
endif


# [keep=<0|1>] Keep intermediate compilation artifacts option
ifeq ($(keep), 1)
	NVCCFLAGS += -keep
endif

# [debug=<0|1>] Generate debug mode code
ifeq ($(debug), 1)
	NVCCFLAGS += -G
endif


#-------------------------------------------------------------------------------
# Compiler and compilation platform
#-------------------------------------------------------------------------------

CUB_DIR = $(dir $(lastword $(MAKEFILE_LIST)))

NVCC = "$(shell which nvcc)"
ifdef nvccver
    NVCC_VERSION = $(nvccver)
else
    NVCC_VERSION = $(strip $(shell nvcc --version | grep release | sed 's/.*release //' |  sed 's/,.*//'))
endif

# detect OS
OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])

# Default flags: verbose kernel properties (regs, smem, cmem, etc.); runtimes for compilation phases 
NVCCFLAGS += $(SM_DEF) -Xptxas -v -Xcudafe -\# 

ifeq (WIN_NT, $(findstring WIN_NT, $(OSUPPER)))
    # For MSVC
    # Enable more warnings and treat as errors
    NVCCFLAGS += -Xcompiler /W3 -Xcompiler /WX
    # Disable excess x86 floating point precision that can lead to results being labeled incorrectly
    NVCCFLAGS += -Xcompiler /fp:strict
    # Help the compiler/linker work with huge numbers of kernels on Windows
	NVCCFLAGS += -Xcompiler /bigobj -Xcompiler /Zm500
	CC = cl
	
	# Multithreaded runtime
	NVCCFLAGS += -Xcompiler /MT
	
ifneq ($(force32), 1)
	CUDART_CYG = "$(shell dirname $(NVCC))/../lib/Win32/cudart.lib"
else
	CUDART_CYG = "$(shell dirname $(NVCC))/../lib/x64/cudart.lib"
endif
	CUDART = "$(shell cygpath -w $(CUDART_CYG))"
else
    # For g++
    # Disable excess x86 floating point precision that can lead to results being labeled incorrectly
    NVCCFLAGS += -Xcompiler -ffloat-store
    CC = g++
ifneq ($(force32), 1)
    CUDART = "$(shell dirname $(NVCC))/../lib/libcudart_static.a"
else
    CUDART = "$(shell dirname $(NVCC))/../lib64/libcudart_static.a"
endif
endif

# Suffix to append to each binary
BIN_SUFFIX = sm$(SM_ARCH)_$(PTX_SUFFIX)_$(NVCC_VERSION)_$(ABI_SUFFIX)_$(CDP_SUFFIX)_$(CPU_ARCH_SUFFIX)


#-------------------------------------------------------------------------------
# Dependency Lists
#-------------------------------------------------------------------------------

rwildcard=$(foreach d,$(wildcard $1*),$(call rwildcard,$d/,$2) $(filter $(subst *,%,$2),$d))

CUB_DEPS = 	$(call rwildcard, $(CUB_DIR),*.cuh) \
			$(CUB_DIR)common.mk
		
