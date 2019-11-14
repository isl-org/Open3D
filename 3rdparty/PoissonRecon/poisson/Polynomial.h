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

#ifndef POLYNOMIAL_INCLUDED
#define POLYNOMIAL_INCLUDED

template< int Degree >
class Polynomial
{
public:
	double coefficients[Degree+1];

	Polynomial( void );
	template< int Degree2 > Polynomial( const Polynomial< Degree2 >& P );
	double operator()( double t ) const;
	double integral( double tMin , double tMax ) const;

	int operator == (const Polynomial& p) const;
	int operator != (const Polynomial& p) const;
	int isZero(void) const;
	void setZero(void);

	template<int Degree2>
	Polynomial& operator  = (const Polynomial<Degree2> &p);
	Polynomial& operator += (const Polynomial& p);
	Polynomial& operator -= (const Polynomial& p);
	Polynomial  operator -  (void) const;
	Polynomial  operator +  (const Polynomial& p) const;
	Polynomial  operator -  (const Polynomial& p) const;
	template<int Degree2>
	Polynomial<Degree+Degree2>  operator *  (const Polynomial<Degree2>& p) const;

	Polynomial& operator += ( double s );
	Polynomial& operator -= ( double s );
	Polynomial& operator *= ( double s );
	Polynomial& operator /= ( double s );
	Polynomial  operator +  ( double s ) const;
	Polynomial  operator -  ( double s ) const;
	Polynomial  operator *  ( double s ) const;
	Polynomial  operator /  ( double s ) const;

	Polynomial scale( double s ) const;
	Polynomial shift( double t ) const;

	template< int _Degree=Degree >
	typename std::enable_if< (_Degree==0) , Polynomial< Degree   > >::type derivative( void ) const { return Polynomial< Degree >(); }
	template< int _Degree=Degree >
	typename std::enable_if< (_Degree> 0) , Polynomial< Degree-1 > >::type derivative( void ) const
	{
		Polynomial< Degree-1 > p;
		for( int i=0 ; i<Degree ; i++ ) p.coefficients[i] = coefficients[i+1]*(i+1);
		return p;
	}
	Polynomial< Degree+1 > integral(void) const;

	void printnl( void ) const;

	Polynomial& addScaled(const Polynomial& p,double scale);
	static void Negate(const Polynomial& in,Polynomial& out);
	static void Subtract(const Polynomial& p1,const Polynomial& p2,Polynomial& q);
	static void Scale(const Polynomial& p,double w,Polynomial& q);
	static void AddScaled(const Polynomial& p1,double w1,const Polynomial& p2,double w2,Polynomial& q);
	static void AddScaled(const Polynomial& p1,const Polynomial& p2,double w2,Polynomial& q);
	static void AddScaled(const Polynomial& p1,double w1,const Polynomial& p2,Polynomial& q);

	int getSolutions( double c , double* roots , double EPS ) const;

	// [NOTE] Both of these methods define the indexing according to DeBoor's algorithm, so that
	// Polynomial< Degree >BSplineComponent( 0 )( 1.0 )=0 for all Degree>0.
	static Polynomial BSplineComponent( int i );
	static void BSplineComponentValues( double x , double* values );
	static void BinomialCoefficients( int bCoefficients[Degree+1] );
};

#include "Polynomial.inl"
#endif // POLYNOMIAL_INCLUDED
