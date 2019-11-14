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

//////////////////
// FunctionData //
//////////////////
template<int Degree,class Real>
FunctionData<Degree,Real>::FunctionData(void)
{
	dotTable=dDotTable=d2DotTable=NULL;
	valueTables=dValueTables=NULL;
	res=0;
}

template<int Degree,class Real>
FunctionData<Degree,Real>::~FunctionData(void)
{
	if(res)
	{
		if(  dotTable) delete[]   dotTable;
		if( dDotTable) delete[]  dDotTable;
		if(d2DotTable) delete[] d2DotTable;
		if( valueTables) delete[]  valueTables;
		if(dValueTables) delete[] dValueTables;
	}
	dotTable=dDotTable=d2DotTable=NULL;
	valueTables=dValueTables=NULL;
	res=0;
}

template<int Degree,class Real>
#if BOUNDARY_CONDITIONS
void FunctionData<Degree,Real>::set( const int& maxDepth , const PPolynomial<Degree>& F , const int& normalize , bool useDotRatios , bool reflectBoundary )
#else // !BOUNDARY_CONDITIONS
void FunctionData<Degree,Real>::set(const int& maxDepth,const PPolynomial<Degree>& F,const int& normalize , bool useDotRatios )
#endif // BOUNDARY_CONDITIONS
{
	this->normalize       = normalize;
	this->useDotRatios    = useDotRatios;
#if BOUNDARY_CONDITIONS
	this->reflectBoundary = reflectBoundary;
#endif // BOUNDARY_CONDITIONS

	depth = maxDepth;
	res = BinaryNode<double>::CumulativeCenterCount( depth );
	res2 = (1<<(depth+1))+1;
	baseFunctions = new PPolynomial<Degree+1>[res];
	// Scale the function so that it has:
	// 0] Value 1 at 0
	// 1] Integral equal to 1
	// 2] Square integral equal to 1
	switch( normalize )
	{
		case 2:
			baseFunction=F/sqrt((F*F).integral(F.polys[0].start,F.polys[F.polyCount-1].start));
			break;
		case 1:
			baseFunction=F/F.integral(F.polys[0].start,F.polys[F.polyCount-1].start);
			break;
		default:
			baseFunction=F/F(0);
	}
	dBaseFunction = baseFunction.derivative();
#if BOUNDARY_CONDITIONS
	leftBaseFunction   = baseFunction + baseFunction.shift( -1 );
	rightBaseFunction  = baseFunction + baseFunction.shift(  1 );
	dLeftBaseFunction  =  leftBaseFunction.derivative();
	dRightBaseFunction = rightBaseFunction.derivative();
#endif // BOUNDARY_CONDITIONS
	double c1,w1;
	for( int i=0 ; i<res ; i++ )
	{
		BinaryNode< double >::CenterAndWidth( i , c1 , w1 );
#if BOUNDARY_CONDITIONS
		if( reflectBoundary )
		{
			int d , off;
			BinaryNode< double >::DepthAndOffset( i , d , off );
			if     ( off==0          ) baseFunctions[i] =  leftBaseFunction.scale( w1 ).shift( c1 );
			else if( off==((1<<d)-1) ) baseFunctions[i] = rightBaseFunction.scale( w1 ).shift( c1 );
			else                       baseFunctions[i] =      baseFunction.scale( w1 ).shift( c1 );
		}
		else baseFunctions[i] = baseFunction.scale(w1).shift(c1);
#else // !BOUNDARY_CONDITIONS
		baseFunctions[i] = baseFunction.scale(w1).shift(c1);
#endif // BOUNDARY_CONDITIONS
		// Scale the function so that it has L2-norm equal to one
		switch( normalize )
		{
			case 2:
				baseFunctions[i]/=sqrt(w1);
				break;
			case 1:
				baseFunctions[i]/=w1;
				break;
		}
	}
}
template<int Degree,class Real>
void FunctionData<Degree,Real>::setDotTables( const int& flags )
{
	clearDotTables( flags );
	int size;
	size = ( res*res + res )>>1;
	if( flags & DOT_FLAG )
	{
		dotTable = new Real[size];
		memset( dotTable , 0 , sizeof(Real)*size );
	}
	if( flags & D_DOT_FLAG )
	{
		dDotTable = new Real[size];
		memset( dDotTable , 0 , sizeof(Real)*size );
	}
	if( flags & D2_DOT_FLAG )
	{
		d2DotTable = new Real[size];
		memset( d2DotTable , 0 , sizeof(Real)*size );
	}
	double t1 , t2;
	t1 = baseFunction.polys[0].start;
	t2 = baseFunction.polys[baseFunction.polyCount-1].start;
	for( int i=0 ; i<res ; i++ )
	{
		double c1 , c2 , w1 , w2;
		BinaryNode<double>::CenterAndWidth( i , c1 , w1 );
#if BOUNDARY_CONDITIONS
		int d1 , d2 , off1 , off2;
		BinaryNode< double >::DepthAndOffset( i , d1 , off1 );
		int boundary1 = 0;
		if     ( reflectBoundary && off1==0             ) boundary1 = -1;
		else if( reflectBoundary && off1==( (1<<d1)-1 ) ) boundary1 =  1;
#endif // BOUNDARY_CONDITIONS
		double start1 = t1 * w1 + c1;
		double end1   = t2 * w1 + c1;
		for( int j=0 ; j<=i ; j++ )
		{
			BinaryNode<double>::CenterAndWidth( j , c2 , w2 );
#if BOUNDARY_CONDITIONS
			BinaryNode< double >::DepthAndOffset( j , d2 , off2 );
			int boundary2 = 0;
			if     ( reflectBoundary && off2==0             ) boundary2 = -1;
			else if( reflectBoundary && off2==( (1<<d2)-1 ) ) boundary2 =  1;
#endif // BOUNDARY_CONDITIONS
			int idx = SymmetricIndex( i , j );

			double start = t1 * w2 + c2;
			double end   = t2 * w2 + c2;
#if BOUNDARY_CONDITIONS
			if( reflectBoundary )
			{
				if( start<0 ) start = 0;
				if( start>1 ) start = 1;
				if( end  <0 )   end = 0;
				if( end  >1 )   end = 1;
			}
#endif // BOUNDARY_CONDITIONS

			if( start<  start1 ) start = start1;
			if( end  >  end1   )   end = end1;
			if( start>= end    ) continue;

#if BOUNDARY_CONDITIONS
			Real dot = dotProduct( c1 , w1 , c2 , w2 , boundary1 , boundary2 );
#else // !BOUNDARY_CONDITIONS
			Real dot = dotProduct( c1 , w1 , c2 , w2 );
#endif // BOUNDARY_CONDITIONS
			if( fabs(dot)<1e-15 ) continue;
			if( flags & DOT_FLAG ) dotTable[idx]=dot;
			if( useDotRatios )
			{
#if BOUNDARY_CONDITIONS
				if( flags &  D_DOT_FLAG )  dDotTable[idx] = -dDotProduct( c1 , w1 , c2 , w2 , boundary1 , boundary2 ) / dot;
				if( flags & D2_DOT_FLAG ) d2DotTable[idx] = d2DotProduct( c1 , w1 , c2 , w2 , boundary1 , boundary2 ) / dot;
#else // !BOUNDARY_CONDITIONS
				if( flags &  D_DOT_FLAG )  dDotTable[idx] = -dDotProduct(c1,w1,c2,w2)/dot;
				if( flags & D2_DOT_FLAG ) d2DotTable[idx] = d2DotProduct(c1,w1,c2,w2)/dot;
#endif // BOUNDARY_CONDITIONS
			}
			else
			{
#if BOUNDARY_CONDITIONS
				if( flags &  D_DOT_FLAG )  dDotTable[idx] =  dDotProduct( c1 , w1 , c2 , w2 , boundary1 , boundary2 );
				if( flags & D2_DOT_FLAG ) d2DotTable[idx] = d2DotProduct( c1 , w1 , c2 , w2 , boundary1 , boundary2 );
#else // !BOUNDARY_CONDTIONS
				if( flags &  D_DOT_FLAG )  dDotTable[idx] =  dDotProduct(c1,w1,c2,w2);
				if( flags & D2_DOT_FLAG ) d2DotTable[idx] = d2DotProduct(c1,w1,c2,w2);
#endif // BOUNDARY_CONDITIONS
			}
		}
	}
}
template<int Degree,class Real>
void FunctionData<Degree,Real>::clearDotTables( const int& flags )
{
	if((flags & DOT_FLAG) && dotTable)
	{
		delete[] dotTable;
		dotTable=NULL;
	}
	if((flags & D_DOT_FLAG) && dDotTable)
	{
		delete[] dDotTable;
		dDotTable=NULL;
	}
	if((flags & D2_DOT_FLAG) && d2DotTable)
	{
		delete[] d2DotTable;
		d2DotTable=NULL;
	}
}
template<int Degree,class Real>
void FunctionData<Degree,Real>::setValueTables( const int& flags , const double& smooth )
{
	clearValueTables();
	if( flags &   VALUE_FLAG )  valueTables = new Real[res*res2];
	if( flags & D_VALUE_FLAG ) dValueTables = new Real[res*res2];
	PPolynomial<Degree+1> function;
	PPolynomial<Degree>  dFunction;
	for( int i=0 ; i<res ; i++ )
	{
		if(smooth>0)
		{
			function=baseFunctions[i].MovingAverage(smooth);
			dFunction=baseFunctions[i].derivative().MovingAverage(smooth);
		}
		else
		{
			function=baseFunctions[i];
			dFunction=baseFunctions[i].derivative();
		}
		for( int j=0 ; j<res2 ; j++ )
		{
			double x=double(j)/(res2-1);
			if(flags &   VALUE_FLAG){ valueTables[j*res+i]=Real( function(x));}
			if(flags & D_VALUE_FLAG){dValueTables[j*res+i]=Real(dFunction(x));}
		}
	}
}
template<int Degree,class Real>
void FunctionData<Degree,Real>::setValueTables(const int& flags,const double& valueSmooth,const double& normalSmooth){
	clearValueTables();
	if(flags &   VALUE_FLAG){ valueTables=new Real[res*res2];}
	if(flags & D_VALUE_FLAG){dValueTables=new Real[res*res2];}
	PPolynomial<Degree+1> function;
	PPolynomial<Degree>  dFunction;
	for(int i=0;i<res;i++){
		if(valueSmooth>0)	{ function=baseFunctions[i].MovingAverage(valueSmooth);}
		else				{ function=baseFunctions[i];}
		if(normalSmooth>0)	{dFunction=baseFunctions[i].derivative().MovingAverage(normalSmooth);}
		else				{dFunction=baseFunctions[i].derivative();}

		for(int j=0;j<res2;j++){
			double x=double(j)/(res2-1);
			if(flags &   VALUE_FLAG){ valueTables[j*res+i]=Real( function(x));}
			if(flags & D_VALUE_FLAG){dValueTables[j*res+i]=Real(dFunction(x));}
		}
	}
}


template<int Degree,class Real>
void FunctionData<Degree,Real>::clearValueTables(void){
	if( valueTables){delete[]  valueTables;}
	if(dValueTables){delete[] dValueTables;}
	valueTables=dValueTables=NULL;
}

#if BOUNDARY_CONDITIONS
template<int Degree,class Real>
Real FunctionData<Degree,Real>::dotProduct( const double& center1 , const double& width1 , const double& center2 , const double& width2 , int boundary1 , int boundary2 ) const
{
	const PPolynomial< Degree > *b1 , *b2;
	if     ( boundary1==-1 ) b1 = & leftBaseFunction;
	else if( boundary1== 0 ) b1 = &     baseFunction;
	else if( boundary1== 1 ) b1 = &rightBaseFunction;
	if     ( boundary2==-1 ) b2 = & leftBaseFunction;
	else if( boundary2== 0 ) b2 = &     baseFunction;
	else if( boundary2== 1 ) b2 = &rightBaseFunction;
	double r=fabs( baseFunction.polys[0].start );
	switch( normalize )
	{
	case 2:
		return Real(((*b1)*b2->scale(width2/width1).shift((center2-center1)/width1)).integral(-2*r,2*r)*width1/sqrt(width1*width2));
	case 1:
		return Real(((*b1)*b2->scale(width2/width1).shift((center2-center1)/width1)).integral(-2*r,2*r)*width1/(width1*width2));
	default:
		return Real(((*b1)*b2->scale(width2/width1).shift((center2-center1)/width1)).integral(-2*r,2*r)*width1);
	}
}
template<int Degree,class Real>
Real FunctionData<Degree,Real>::dDotProduct( const double& center1 , const double& width1 , const double& center2 , const double& width2 , int boundary1 , int boundary2 ) const
{
	const PPolynomial< Degree-1 > *b1;
	const PPolynomial< Degree   > *b2;
	if     ( boundary1==-1 ) b1 = & dLeftBaseFunction;
	else if( boundary1== 0 ) b1 = &     dBaseFunction;
	else if( boundary1== 1 ) b1 = &dRightBaseFunction;
	if     ( boundary2==-1 ) b2 = &  leftBaseFunction;
	else if( boundary2== 0 ) b2 = &      baseFunction;
	else if( boundary2== 1 ) b2 = & rightBaseFunction;
	double r=fabs(baseFunction.polys[0].start);
	switch(normalize){
		case 2:
			return Real(((*b1)*b2->scale(width2/width1).shift((center2-center1)/width1)).integral(-2*r,2*r)/sqrt(width1*width2));
		case 1:
			return Real(((*b1)*b2->scale(width2/width1).shift((center2-center1)/width1)).integral(-2*r,2*r)/(width1*width2));
		default:
			return Real(((*b1)*b2->scale(width2/width1).shift((center2-center1)/width1)).integral(-2*r,2*r));
	}
}
template<int Degree,class Real>
Real FunctionData<Degree,Real>::d2DotProduct( const double& center1 , const double& width1 , const double& center2 , const double& width2 , int boundary1 , int boundary2 ) const
{
	const PPolynomial< Degree-1 > *b1 , *b2;
	if     ( boundary1==-1 ) b1 = & dLeftBaseFunction;
	else if( boundary1== 0 ) b1 = &     dBaseFunction;
	else if( boundary1== 1 ) b1 = &dRightBaseFunction;
	if     ( boundary2==-1 ) b2 = & dLeftBaseFunction;
	else if( boundary2== 0 ) b2 = &     dBaseFunction;
	else if( boundary2== 1 ) b2 = &dRightBaseFunction;
	double r=fabs(baseFunction.polys[0].start);
	switch( normalize )
	{
		case 2:
			return Real(((*b1)*b2->scale(width2/width1).shift((center2-center1)/width1)).integral(-2*r,2*r)/width2/sqrt(width1*width2));
		case 1:
			return Real(((*b1)*b2->scale(width2/width1).shift((center2-center1)/width1)).integral(-2*r,2*r)/width2/(width1*width2));
		default:
			return Real(((*b1)*b2->scale(width2/width1).shift((center2-center1)/width1)).integral(-2*r,2*r)/width2);
	}
}
#else // !BOUNDARY_CONDITIONS
template<int Degree,class Real>
Real FunctionData<Degree,Real>::dotProduct(const double& center1,const double& width1,const double& center2,const double& width2) const{
	double r=fabs(baseFunction.polys[0].start);
	switch( normalize )
	{
		case 2:
			return Real((baseFunction*baseFunction.scale(width2/width1).shift((center2-center1)/width1)).integral(-2*r,2*r)*width1/sqrt(width1*width2));
		case 1:
			return Real((baseFunction*baseFunction.scale(width2/width1).shift((center2-center1)/width1)).integral(-2*r,2*r)*width1/(width1*width2));
		default:
			return Real((baseFunction*baseFunction.scale(width2/width1).shift((center2-center1)/width1)).integral(-2*r,2*r)*width1);
	}
}
template<int Degree,class Real>
Real FunctionData<Degree,Real>::dDotProduct(const double& center1,const double& width1,const double& center2,const double& width2) const{
	double r=fabs(baseFunction.polys[0].start);
	switch(normalize){
		case 2:
			return Real((dBaseFunction*baseFunction.scale(width2/width1).shift((center2-center1)/width1)).integral(-2*r,2*r)/sqrt(width1*width2));
		case 1:
			return Real((dBaseFunction*baseFunction.scale(width2/width1).shift((center2-center1)/width1)).integral(-2*r,2*r)/(width1*width2));
		default:
			return Real((dBaseFunction*baseFunction.scale(width2/width1).shift((center2-center1)/width1)).integral(-2*r,2*r));
	}
}
template<int Degree,class Real>
Real FunctionData<Degree,Real>::d2DotProduct(const double& center1,const double& width1,const double& center2,const double& width2) const{
	double r=fabs(baseFunction.polys[0].start);
	switch(normalize){
		case 2:
			return Real((dBaseFunction*dBaseFunction.scale(width2/width1).shift((center2-center1)/width1)).integral(-2*r,2*r)/width2/sqrt(width1*width2));
		case 1:
			return Real((dBaseFunction*dBaseFunction.scale(width2/width1).shift((center2-center1)/width1)).integral(-2*r,2*r)/width2/(width1*width2));
		default:
			return Real((dBaseFunction*dBaseFunction.scale(width2/width1).shift((center2-center1)/width1)).integral(-2*r,2*r)/width2);
	}
}
#endif // BOUNDARY_CONDITIONS
template<int Degree,class Real>
inline int FunctionData<Degree,Real>::SymmetricIndex( const int& i1 , const int& i2 )
{
	if( i1>i2 ) return ((i1*i1+i1)>>1)+i2;
	else        return ((i2*i2+i2)>>1)+i1;
}
template<int Degree,class Real>
inline int FunctionData<Degree,Real>::SymmetricIndex( const int& i1 , const int& i2 , int& index )
{
	if( i1<i2 )
	{
		index = ((i2*i2+i2)>>1)+i1;
		return 1;
	}
	else{
		index = ((i1*i1+i1)>>1)+i2;
		return 0;
	}
}
