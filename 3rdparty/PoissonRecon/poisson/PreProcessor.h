/*
Copyright (c) 2019, Michael Kazhdan and Matthew Bolitho
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer. Redistributions in binary form must reproduce
the above copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the distribution. 

Neither the name of the Johns Hopkins University nor the names of its contributors
may be used to endorse or promote products derived from this software without specific
prior written permission. 

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO THE IMPLIED WARRANTIES 
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE  GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
*/

#ifndef PRE_PROCESSOR_INCLUDED
#define PRE_PROCESSOR_INCLUDED

#undef BIG_DATA									// Supports processing requiring more than 32-bit integers for indexing
												// Note: enabling BIG_DATA can generate .ply files using "longlong" for vertex indices instead of "int".
												// These are not standardly supported by .ply reading/writing applications.
												// The executable ChunkPLY can help by partitioning the mesh into more manageable chunks
												// (each of which is small enough to be represented using 32-bit indexing.)
						
#undef FAST_COMPILE								// If enabled, only a single version of the code is compiled
#undef SHOW_WARNINGS							// Display compilation warnings
#undef ARRAY_DEBUG								// If enabled, array access is tested for validity
#undef USE_SEG_FAULT_HANDLER					// Tries to dump a stack trace in the case of a segfault (gcc only)

#ifdef BIG_DATA
#define USE_DEEP_TREE_NODES						// Chances are that if you are using big data, you want to support a tree with depth>15.
#endif // BIG_DATA

#define VERSION "12.00"							// The version of the code
#define MEMORY_ALLOCATOR_BLOCK_SIZE 1<<12		// The chunk size for memory allocation

#endif // PRE_PROCESSOR_INCLUDED