// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2015 Qianyi Zhou <Qianyi.Zhou@gmail.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include <stdio.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define NUM_THREADS 4
#define NUM_START 1
#define NUM_END 10

int main()
{
	int i, nRet = 0, nSum = 0, nStart = NUM_START, nEnd = NUM_END;
	int nThreads = 1, nTmp = nStart + nEnd;
	unsigned uTmp = (unsigned(nEnd - nStart + 1) *
			unsigned(nTmp)) / 2;
	int nSumCalc = uTmp;

	if (nTmp < 0) {
		nSumCalc = -nSumCalc;
	}

#ifdef _OPENMP
	printf("OpenMP is supported.\n");
#else
	printf("OpenMP is not supported.\n");
#endif

#ifdef _OPENMP
	omp_set_num_threads(NUM_THREADS);
#endif

#pragma omp parallel default(none) private(i) shared(nSum, nThreads, nStart, nEnd)
	{
#ifdef _OPENMP
#pragma omp master
		nThreads = omp_get_num_threads();
#endif

#pragma omp for
		for (i = nStart; i <= nEnd; ++i) {
#pragma omp atomic
			nSum += i;
		}
	}

	if (nThreads == NUM_THREADS) {
		printf("%d OpenMP threads were used.\n", NUM_THREADS);
		nRet = 0;
	} else {
		printf("Expected %d OpenMP threads, but %d were used.\n",
				NUM_THREADS, nThreads);
		nRet = 1;
	}

	if (nSum != nSumCalc) {
		printf("The sum of %d through %d should be %d, "
				"but %d was reported!\n",
				NUM_START, NUM_END, nSumCalc, nSum);
		nRet = 1;
	} else
		printf("The sum of %d through %d is %d\n",
				NUM_START, NUM_END, nSum);
}
