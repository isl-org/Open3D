##===-- Makefile ----------------------------------------------------------===##
#
# Copyright (C) 2017-2019 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# This file incorporates work covered by the following copyright and permission
# notice:
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
#
##===----------------------------------------------------------------------===##

# GNU Makefile that builds and runs example.
run_cmd=
PROG=convex_hull.exe
ARGS=

CXXFLAGS += -D__PSTL_USE_TBB -std=c++11

# Set by default icc as C++ compiler if it's present
ifneq (,$(shell which icc 2>/dev/null))
CXX = icc
endif # which icc

ifeq ($(CXX),icc)
CXXFLAGS += -qopenmp-simd
ifneq (, $(filter $(target), mic))
CXXFLAGS += -mmic
else
CXXFLAGS += -xHOST
endif # target is mic or host?
endif # icc?

ifeq ($(shell uname), Linux)
LIBS+= -lrt
else ifeq ($(shell uname), Darwin)
override CXXFLAGS += -Wl,-rpath,$(TBBROOT)/lib
endif

all:    release test

release: *.cpp
	$(CXX) -O2 -DNDEBUG $(CXXFLAGS) -o $(PROG) $^ -ltbb $(LIBS)

debug: *.cpp
	$(CXX) -O0 -g -DTBB_USE_DEBUG=1 $(CXXFLAGS) -o $(PROG) $^ -ltbb_debug $(LIBS)

clean:
	$(RM) $(PROG) *.o *.d

test:
	$(run_cmd) ./$(PROG) $(ARGS)

perf_build: release

perf_run: test
