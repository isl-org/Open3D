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

#ifndef FACTOR_INCLUDED
#define FACTOR_INCLUDED

#include <math.h>
#include <complex>
#ifndef SQRT_3
#define SQRT_3 1.7320508075688772935
#endif // SQRT_3
inline int Factor( double a1 , double a0 , std::complex< double > roots[1] , double EPS )
{
	if( fabs(a1)<=EPS ) return 0;
	roots[0] = std::complex< double >( -a0/a1 , 0 );
	return 1;
}
inline int Factor( double a2 , double a1 , double a0 , std::complex< double > roots[2] , double EPS )
{
	double d;
	if( fabs(a2)<=EPS ) return Factor( a1 , a0 , roots , EPS );

	d = a1*a1 - 4*a0*a2;
	a1 /= (2*a2);
	if( d<0 )
	{
		d=sqrt(-d)/(2*a2);
		roots[0] = std::complex< double >( -a1 , -d );
		roots[1] = std::complex< double >( -a1 ,  d );
	}
	else
	{
		d = sqrt(d)/(2*a2);
		roots[0] = std::complex< double >( -a1-d , 0 );
		roots[1] = std::complex< double >( -a1+d , 0 );
	}
	return 2;
}
// Solution taken from: http://mathworld.wolfram.com/CubicFormula.html
// and http://www.csit.fsu.edu/~burkardt/f_src/subpak/subpak.f90
inline int Factor( double a3 , double a2 , double a1 , double a0 , std::complex< double > roots[3] , double EPS )
{
	double q,r,r2,q3;

	if( fabs(a3)<=EPS ) return Factor( a2 , a1 , a0 , roots , EPS );
	a2 /= a3 , a1 /= a3 , a0 /= a3;

	q = -(3*a1-a2*a2)/9;
	r = -(9*a2*a1-27*a0-2*a2*a2*a2)/54;
	r2 = r*r;
	q3 = q*q*q;

	if(r2<q3)
	{
		double sqrQ = sqrt(q);
		double theta = acos ( r / (sqrQ*q) );
		double cTheta=cos(theta/3)*sqrQ;
		double sTheta=sin(theta/3)*sqrQ*SQRT_3/2;
		roots[0] = std::complex< double >( -2*cTheta , 0 );
		roots[1] = std::complex< double >( -2*(-cTheta*0.5-sTheta) , 0 );
		roots[2] = std::complex< double >( -2*(-cTheta*0.5+sTheta) , 0 );
	}
	else
	{
		double t , s1 , s2 , sqr=sqrt(r2-q3);
		t = -r+sqr;
		if(t<0) s1 = -pow( -t , 1.0/3 );
		else    s1 =  pow(  t , 1.0/3 );
		t = -r-sqr;
		if( t<0 ) s2 = -pow( -t , 1.0/3 );
		else      s2 =  pow(  t , 1.0/3 );
		roots[0] = std::complex< double >( s1+s2 , 0 );
		s1 /= 2 , s2 /= 2;
		roots[1] = std::complex< double >( -s1-s2 ,  SQRT_3*(s1-s2) );
		roots[2] = std::complex< double >( -s1-s2 , -SQRT_3*(s1-s2) );
	}
	roots[0] -= a2/3;
	roots[1] -= a2/3;
	roots[2] -= a2/3;
	return 3;
}
// Solution taken from: http://mathworld.wolfram.com/QuarticEquation.html
// and http://www.csit.fsu.edu/~burkardt/f_src/subpak/subpak.f90
inline int Factor( double a4 , double a3 , double a2 , double a1 , double a0 , std::complex< double > roots[4] , double EPS )
{
	std::complex< double > R , D , E , R2;

	if( fabs(a4)<EPS ) return Factor( a3 , a2 , a1 , a0 , roots , EPS );
	a3 /= a4 , a2 /= a4 , a1 /= a4 , a0 /= a4;

	Factor( 1.0 , -a2 , a3*a1-4.0*a0 , -a3*a3*a0+4.0*a2*a0-a1*a1 , roots , EPS );

	R2 = std::complex< double >( a3*a3/4.0-a2+roots[0].real() , 0 );
	R = sqrt( R2 );
	if( fabs( R.real() )>10e-8 )
	{
		std::complex< double > temp1 , temp2 , p1 , p2;

		p1 = std::complex< double >( a3*a3*0.75-2.0*a2-R2.real() , 0 );

		temp2 = std::complex< double >( (4.0*a3*a2-8.0*a1-a3*a3*a3)/4.0 , 0 );
		p2 = temp2 / R;
		temp1 = p1+p2;
		temp2 = p1-p2;
		D = sqrt( temp1 );
		E = sqrt( temp2 );
	}
	else
	{
		R = std::complex< double >( 0 , 0 );
		std::complex< double > temp1 , temp2;
		temp1 = std::complex< double >( roots[0].real()*roots[0].real()-4.0*a0 , 0 );
		temp2 = sqrt( temp1 );

		temp1 = std::complex< double >( a3*a3*0.75-2.0*a2+2.0*temp2.real() ,  2.0*temp2.imag() );
		D = sqrt( temp1 );

		temp1 = std::complex< double >( a3*a3*0.75-2.0*a2-2.0*temp2.real() , -2.0*temp2.imag() );
		E = sqrt( temp1 );
	}

	roots[0] =  R/2. + D/2. - a3/4;
	roots[1] =  R/2. - D/2. - a3/4;
	roots[2] = -R/2. + E/2. - a3/4;
	roots[3] = -R/2. - E/2. - a3/4;

	return 4;
}
#endif // FACTOR_INCLUDED