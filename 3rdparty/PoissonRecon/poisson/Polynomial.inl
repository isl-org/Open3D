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

#include <float.h>
#include <math.h>
#include <algorithm>
#include "Factor.h"

////////////////
// Polynomial //
////////////////
template<int Degree>
Polynomial<Degree>::Polynomial(void){memset(coefficients,0,sizeof(double)*(Degree+1));}
template<int Degree>
template<int Degree2>
Polynomial<Degree>::Polynomial(const Polynomial<Degree2>& P){
	memset(coefficients,0,sizeof(double)*(Degree+1));
	for(int i=0;i<=Degree && i<=Degree2;i++){coefficients[i]=P.coefficients[i];}
}


template<int Degree>
template<int Degree2>
Polynomial<Degree>& Polynomial<Degree>::operator  = (const Polynomial<Degree2> &p){
	int d=Degree<Degree2?Degree:Degree2;
	memset(coefficients,0,sizeof(double)*(Degree+1));
	memcpy(coefficients,p.coefficients,sizeof(double)*(d+1));
	return *this;
}

template<int Degree>
Polynomial<Degree+1> Polynomial<Degree>::integral(void) const{
	Polynomial<Degree+1> p;
	p.coefficients[0]=0;
	for(int i=0;i<=Degree;i++){p.coefficients[i+1]=coefficients[i]/(i+1);}
	return p;
}
template< > double Polynomial< 0 >::operator() ( double t ) const { return coefficients[0]; }
template< > double Polynomial< 1 >::operator() ( double t ) const { return coefficients[0]+coefficients[1]*t; }
template< > double Polynomial< 2 >::operator() ( double t ) const { return coefficients[0]+(coefficients[1]+coefficients[2]*t)*t; }
template< int Degree >
double Polynomial<Degree>::operator() ( double t ) const{
	double v=coefficients[Degree];
	for( int d=Degree-1 ; d>=0 ; d-- ) v = v*t + coefficients[d];
	return v;
}
template<int Degree>
double Polynomial<Degree>::integral( double tMin , double tMax ) const
{
	double v=0;
	double t1,t2;
	t1=tMin;
	t2=tMax;
	for(int i=0;i<=Degree;i++){
		v+=coefficients[i]*(t2-t1)/(i+1);
		if(t1!=-DBL_MAX && t1!=DBL_MAX){t1*=tMin;}
		if(t2!=-DBL_MAX && t2!=DBL_MAX){t2*=tMax;}
	}
	return v;
}
template<int Degree>
int Polynomial<Degree>::operator == (const Polynomial& p) const{
	for(int i=0;i<=Degree;i++){if(coefficients[i]!=p.coefficients[i]){return 0;}}
	return 1;
}
template<int Degree>
int Polynomial<Degree>::operator != (const Polynomial& p) const{
	for(int i=0;i<=Degree;i++){if(coefficients[i]==p.coefficients[i]){return 0;}}
	return 1;
}
template<int Degree>
int Polynomial<Degree>::isZero(void) const{
	for(int i=0;i<=Degree;i++){if(coefficients[i]!=0){return 0;}}
	return 1;
}
template<int Degree>
void Polynomial<Degree>::setZero(void){memset(coefficients,0,sizeof(double)*(Degree+1));}

template<int Degree>
Polynomial<Degree>& Polynomial<Degree>::addScaled(const Polynomial& p,double s){
	for(int i=0;i<=Degree;i++){coefficients[i]+=p.coefficients[i]*s;}
	return *this;
}
template<int Degree>
Polynomial<Degree>& Polynomial<Degree>::operator += (const Polynomial<Degree>& p){
	for(int i=0;i<=Degree;i++){coefficients[i]+=p.coefficients[i];}
	return *this;
}
template<int Degree>
Polynomial<Degree>& Polynomial<Degree>::operator -= (const Polynomial<Degree>& p){
	for(int i=0;i<=Degree;i++){coefficients[i]-=p.coefficients[i];}
	return *this;
}
template<int Degree>
Polynomial<Degree> Polynomial<Degree>::operator + (const Polynomial<Degree>& p) const{
	Polynomial q;
	for(int i=0;i<=Degree;i++){q.coefficients[i]=(coefficients[i]+p.coefficients[i]);}
	return q;
}
template<int Degree>
Polynomial<Degree> Polynomial<Degree>::operator - (const Polynomial<Degree>& p) const{
	Polynomial q;
	for(int i=0;i<=Degree;i++)	{q.coefficients[i]=coefficients[i]-p.coefficients[i];}
	return q;
}
template<int Degree>
void Polynomial<Degree>::Scale(const Polynomial& p,double w,Polynomial& q){
	for(int i=0;i<=Degree;i++){q.coefficients[i]=p.coefficients[i]*w;}
}
template<int Degree>
void Polynomial<Degree>::AddScaled(const Polynomial& p1,double w1,const Polynomial& p2,double w2,Polynomial& q){
	for(int i=0;i<=Degree;i++){q.coefficients[i]=p1.coefficients[i]*w1+p2.coefficients[i]*w2;}
}
template<int Degree>
void Polynomial<Degree>::AddScaled(const Polynomial& p1,double w1,const Polynomial& p2,Polynomial& q){
	for(int i=0;i<=Degree;i++){q.coefficients[i]=p1.coefficients[i]*w1+p2.coefficients[i];}
}
template<int Degree>
void Polynomial<Degree>::AddScaled(const Polynomial& p1,const Polynomial& p2,double w2,Polynomial& q){
	for(int i=0;i<=Degree;i++){q.coefficients[i]=p1.coefficients[i]+p2.coefficients[i]*w2;}
}

template<int Degree>
void Polynomial<Degree>::Subtract(const Polynomial &p1,const Polynomial& p2,Polynomial& q){
	for(int i=0;i<=Degree;i++){q.coefficients[i]=p1.coefficients[i]-p2.coefficients[i];}
}
template<int Degree>
void Polynomial<Degree>::Negate(const Polynomial& in,Polynomial& out){
	out=in;
	for(int i=0;i<=Degree;i++){out.coefficients[i]=-out.coefficients[i];}
}

template<int Degree>
Polynomial<Degree> Polynomial<Degree>::operator - (void) const{
	Polynomial q=*this;
	for(int i=0;i<=Degree;i++){q.coefficients[i]=-q.coefficients[i];}
	return q;
}
template<int Degree>
template<int Degree2>
Polynomial<Degree+Degree2> Polynomial<Degree>::operator * (const Polynomial<Degree2>& p) const{
	Polynomial<Degree+Degree2> q;
	for(int i=0;i<=Degree;i++){for(int j=0;j<=Degree2;j++){q.coefficients[i+j]+=coefficients[i]*p.coefficients[j];}}
	return q;
}

template<int Degree>
Polynomial<Degree>& Polynomial<Degree>::operator += ( double s )
{
	coefficients[0]+=s;
	return *this;
}
template<int Degree>
Polynomial<Degree>& Polynomial<Degree>::operator -= ( double s )
{
	coefficients[0]-=s;
	return *this;
}
template<int Degree>
Polynomial<Degree>& Polynomial<Degree>::operator *= ( double s )
{
	for(int i=0;i<=Degree;i++){coefficients[i]*=s;}
	return *this;
}
template<int Degree>
Polynomial<Degree>& Polynomial<Degree>::operator /= ( double s )
{
	for(int i=0;i<=Degree;i++){coefficients[i]/=s;}
	return *this;
}
template<int Degree>
Polynomial<Degree> Polynomial<Degree>::operator + ( double s ) const
{
	Polynomial<Degree> q=*this;
	q.coefficients[0]+=s;
	return q;
}
template<int Degree>
Polynomial<Degree> Polynomial<Degree>::operator - ( double s ) const
{
	Polynomial q=*this;
	q.coefficients[0]-=s;
	return q;
}
template<int Degree>
Polynomial<Degree> Polynomial<Degree>::operator * ( double s ) const
{
	Polynomial q;
	for(int i=0;i<=Degree;i++){q.coefficients[i]=coefficients[i]*s;}
	return q;
}
template<int Degree>
Polynomial<Degree> Polynomial<Degree>::operator / ( double s ) const
{
	Polynomial q;
	for( int i=0 ; i<=Degree ; i++ ) q.coefficients[i] = coefficients[i]/s;
	return q;
}
template<int Degree>
Polynomial<Degree> Polynomial<Degree>::scale( double s ) const
{
	Polynomial q=*this;
	double s2=1.0;
	for(int i=0;i<=Degree;i++){
		q.coefficients[i]*=s2;
		s2/=s;
	}
	return q;
}
template<int Degree>
Polynomial<Degree> Polynomial<Degree>::shift( double t ) const
{
	Polynomial<Degree> q;
	for(int i=0;i<=Degree;i++){
		double temp=1;
		for(int j=i;j>=0;j--){
			q.coefficients[j]+=coefficients[i]*temp;
			temp*=-t*j;
			temp/=(i-j+1);
		}
	}
	return q;
}
template<int Degree>
void Polynomial<Degree>::printnl(void) const{
	for(int j=0;j<=Degree;j++){
		printf("%6.4f x^%d ",coefficients[j],j);
		if(j<Degree && coefficients[j+1]>=0){printf("+");}
	}
	printf("\n");
}
template< int Degree >
int Polynomial<Degree>::getSolutions( double c , double* roots , double EPS ) const
{
	std::complex< double > _roots[4];
	int _rCount=0;
	switch( Degree )
	{
		case 1: _rCount = Factor(                                                       coefficients[1] , coefficients[0]-c , _roots , EPS ) ; break;
		case 2:	_rCount = Factor(                                     coefficients[2] , coefficients[1] , coefficients[0]-c , _roots , EPS ) ; break;
		case 3: _rCount = Factor(                   coefficients[3] , coefficients[2] , coefficients[1] , coefficients[0]-c , _roots , EPS ) ; break;
//		case 4: _rCount = Factor( coefficients[4] , coefficients[3] , coefficients[2] , coefficients[1] , coefficients[0]-c , _roots , EPS ) ; break;
		default: printf( "Can't solve polynomial of degree: %d\n" , Degree );
	}
	int rCount = 0;
	for( int i=0 ; i<_rCount ; i++ ) if( fabs( _roots[i].imag() )<=EPS ) roots[rCount++] = _roots[i].real();
	return rCount;
}
// The 0-th order B-spline
template< >
Polynomial< 0 > Polynomial< 0 >::BSplineComponent( int i )
{
	Polynomial p;
	p.coefficients[0] = 1.;
	return p;
}

// The Degree-th order B-spline
template< int Degree >
Polynomial< Degree > Polynomial< Degree >::BSplineComponent( int i )
{
	// B_d^i(x) = \int_x^1 B_{d-1}^{i}(y) dy + \int_0^x B_{d-1}^{i-1} y dy
	//          = \int_0^1 B_{d-1}^{i}(y) dy - \int_0^x B_{d-1}^{i}(y) dy + \int_0^x B_{d-1}^{i-1} y dy
	Polynomial p;
	if( i<Degree )
	{
		Polynomial< Degree > _p = Polynomial< Degree-1 >::BSplineComponent( i ).integral();
		p -= _p;
		p.coefficients[0] += _p(1);
	}
	if( i>0 )
	{
		Polynomial< Degree > _p = Polynomial< Degree-1 >::BSplineComponent( i-1 ).integral();
		p += _p;
	}
	return p;
}


// The 0-th order B-spline values
template< > void Polynomial< 0 >::BSplineComponentValues( double x , double* values ){ values[0] = 1.; }
// The Degree-th order B-spline
template< int Degree > void Polynomial< Degree >::BSplineComponentValues( double x , double* values )
{
	const double Scale = 1./Degree;
	Polynomial< Degree-1 >::BSplineComponentValues( x , values+1 );
	values[0] = values[1] * (1.-x) * Scale;
	for( int i=1 ; i<Degree ; i++ )
	{
		double x1 = (x-i+Degree) , x2 = (-x+i+1);
		values[i] = ( values[i]*x1 + values[i+1]*x2 ) * Scale;
	}
	values[Degree] *= x * Scale;
}

// Using the recurrence formulation for Pascal's triangle
template< > void Polynomial< 0 >::BinomialCoefficients( int bCoefficients[1] ){ bCoefficients[0] = 1; }
template< int Degree > void Polynomial< Degree >::BinomialCoefficients( int bCoefficients[Degree+1] )
{
	Polynomial< Degree-1 >::BinomialCoefficients( bCoefficients );
	int leftValue = 0;
	for( int i=0 ; i<Degree ; i++ )
	{
		int temp = bCoefficients[i];
		bCoefficients[i] += leftValue;
		leftValue = temp;
	}
	bCoefficients[Degree] = 1;
}
