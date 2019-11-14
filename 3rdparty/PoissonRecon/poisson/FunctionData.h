/*
Copyright (c) 2006, Michael Kazhdan and Matthew Bolitho
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

#ifndef FUNCTION_DATA_INCLUDED
#define FUNCTION_DATA_INCLUDED

#define BOUNDARY_CONDITIONS 1


#include "PPolynomial.h"

template<int Degree,class Real>
class FunctionData{
	bool useDotRatios;
	int normalize;
#if BOUNDARY_CONDITIONS
	bool reflectBoundary;
#endif // BOUNDARY_CONDITIONS
public:
	const static int     DOT_FLAG = 1;
	const static int   D_DOT_FLAG = 2;
	const static int  D2_DOT_FLAG = 4;
	const static int   VALUE_FLAG = 1;
	const static int D_VALUE_FLAG = 2;

	int depth , res , res2;
	Real *dotTable , *dDotTable , *d2DotTable;
	Real *valueTables , *dValueTables;
#if BOUNDARY_CONDITIONS
	PPolynomial<Degree> baseFunction , leftBaseFunction , rightBaseFunction;
	PPolynomial<Degree-1> dBaseFunction , dLeftBaseFunction , dRightBaseFunction;
#else // !BOUNDARY_CONDITIONS
	PPolynomial<Degree> baseFunction;
	PPolynomial<Degree-1> dBaseFunction;
#endif // BOUNDARY_CONDITIONS
	PPolynomial<Degree+1>* baseFunctions;

	FunctionData(void);
	~FunctionData(void);

	virtual void   setDotTables(const int& flags);
	virtual void clearDotTables(const int& flags);

	virtual void   setValueTables(const int& flags,const double& smooth=0);
	virtual void   setValueTables(const int& flags,const double& valueSmooth,const double& normalSmooth);
	virtual void clearValueTables(void);

	/********************************************************
	 * Sets the translates and scales of the basis function
	 * up to the prescribed depth
	 * <maxDepth> the maximum depth
	 * <F> the basis function
	 * <normalize> how the functions should be scaled
	 *      0] Value at zero equals 1
	 *      1] Integral equals 1
	 *		2] L2-norm equals 1
	 * <useDotRatios> specifies if dot-products of derivatives
	 * should be pre-divided by function integrals
	 * <reflectBoundary> spcifies if function space should be
	 * forced to be reflectively symmetric across the boundary
	 ********************************************************/
#if BOUNDARY_CONDITIONS
	void set( const int& maxDepth , const PPolynomial<Degree>& F , const int& normalize , bool useDotRatios=true , bool reflectBoundary=false );
#else // !BOUNDARY_CONDITIONS
	void set(const int& maxDepth,const PPolynomial<Degree>& F,const int& normalize , bool useDotRatios=true );
#endif // BOUNDARY_CONDITIONS

#if BOUNDARY_CONDITIONS
	Real   dotProduct( const double& center1 , const double& width1 , const double& center2 , const double& width2 , int boundary1 , int boundary2 ) const;
	Real  dDotProduct( const double& center1 , const double& width1 , const double& center2 , const double& width2 , int boundary1 , int boundary2 ) const;
	Real d2DotProduct( const double& center1 , const double& width1 , const double& center2 , const double& width2 , int boundary1 , int boundary2 ) const;
#else // !BOUNDARY_CONDITIONS
	Real   dotProduct( const double& center1 , const double& width1 , const double& center2 , const double& width2 ) const;
	Real  dDotProduct( const double& center1 , const double& width1 , const double& center2 , const double& width2 ) const;
	Real d2DotProduct( const double& center1 , const double& width1 , const double& center2 , const double& width2 ) const;
#endif // BOUNDARY_CONDITIONS

	static inline int SymmetricIndex( const int& i1 , const int& i2 );
	static inline int SymmetricIndex( const int& i1 , const int& i2 , int& index  );
};


#include "FunctionData.inl"
#endif // FUNCTION_DATA_INCLUDED