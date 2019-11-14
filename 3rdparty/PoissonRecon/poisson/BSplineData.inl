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

///////////////////////////
// BSplineEvaluationData //
///////////////////////////
template< unsigned int FEMSig >
double BSplineEvaluationData< FEMSig >::Value( int depth , int off , double s , int d )
{
	if( s<0 || s>1 ) return 0.;

	int res = 1<<depth;
	if( OutOfBounds( depth , off ) ) return 0;

	typename BSplineData< FEMSig , Degree >::BSplineComponents components( depth , off );

	// [NOTE] This is an ugly way to ensure that when s=1 we evaluate using a B-Spline component within the valid range.
	int ii = std::max< int >( 0 , std::min< int >( res-1 , (int)floor( s * res ) ) ) - off;

	if( ii<BSplineSupportSizes< Degree >::SupportStart || ii>BSplineSupportSizes< Degree >::SupportEnd ) return 0;
	return d<=Degree ? components[ii-BSplineSupportSizes< Degree >::SupportStart][d](s) : 0;
}
template< unsigned int FEMSig >
double BSplineEvaluationData< FEMSig >::Integral( int depth , int off , double b , double e , int d )
{
	double integral = 0;
	// Check for valid integration bounds
	if( OutOfBounds( depth , off ) ) return 0;
	if( b>=e || b>=1 || e<=0 ) return 0;
	if( b<0 ) b=0;
	if( e>1 ) e=1;

	int res = 1<<depth;
	double _b = ( (double)( off     + BSplineSupportSizes< Degree >::SupportStart ) )/res;
	double _e = ( (double)( off + 1 + BSplineSupportSizes< Degree >::SupportEnd   ) )/res;
	if( b>=_e || e<=_b ) return 0;
	typename BSplineData< FEMSig , Degree >::BSplineComponents components( depth , off );
	for( int i=BSplineSupportSizes< Degree >::SupportStart ; i<=BSplineSupportSizes< Degree >::SupportEnd ; i++ )
	{
		// The index of the current cell
		int c = off + i;
		// The bounds of the current cell
		_b = std::max< double >( b , ( (double)c ) / res ) , _e = std::min< double >( e , ( (double)(c+1) )/res );
		if( _b<_e ) integral += d<=Degree ? components[i-BSplineSupportSizes< Degree >::SupportStart][d].integral( _b , _e ) : 0;
	}
	return integral;
}
template< unsigned int FEMSig >
template< unsigned int D >
void BSplineEvaluationData< FEMSig >::SetCenterEvaluator( typename CenterEvaluator< D >::Evaluator& evaluator , int depth )
{
	evaluator._depth = depth;
	int res = 1<<depth;
	for( int i=0 ; i<IndexSize ; i++ ) for( int j=BSplineSupportSizes< Degree >::SupportStart ; j<=BSplineSupportSizes< Degree >::SupportEnd ; j++ )
	{
		int ii = IndexToOffset( depth , i );
		double s = 0.5 + ii + j;
		for( int d1=0 ; d1<=D ; d1++ ) evaluator._ccValues[d1][i][j-BSplineSupportSizes< Degree >::SupportStart] = Value( depth , ii , s/res , d1 );
	}
}
template< unsigned int FEMSig >
template< unsigned int D >
void BSplineEvaluationData< FEMSig >::SetChildCenterEvaluator( typename CenterEvaluator< D >::ChildEvaluator& evaluator , int parentDepth )
{
	evaluator._parentDepth = parentDepth;
	int res = 1<<(parentDepth+1);
	for( int i=0 ; i<IndexSize ; i++ ) for( int j=BSplineSupportSizes< Degree >::ChildSupportStart ; j<=BSplineSupportSizes< Degree >::ChildSupportEnd ; j++ )
	{
		int ii = IndexToOffset( parentDepth , i );
		double s = 0.5 + 2*ii + j;
		for( int d1=0 ; d1<=D ; d1++ ) evaluator._pcValues[d1][i][j-BSplineSupportSizes< Degree >::ChildSupportStart] = Value( parentDepth , ii , s/res , d1 );
	}
}
template< unsigned int FEMSig >
template< unsigned int D >
double BSplineEvaluationData< FEMSig >::CenterEvaluator< D >::Evaluator::value( int fIdx , int cIdx , int d ) const
{
	int dd = cIdx-fIdx , res = 1<<(_depth);
	if( cIdx<0 || cIdx>=res || OutOfBounds( _depth , fIdx ) || dd<BSplineSupportSizes< Degree >::SupportStart || dd>BSplineSupportSizes< Degree >::SupportEnd ) return 0;
	return _ccValues[d][ OffsetToIndex( _depth , fIdx ) ][dd-BSplineSupportSizes< Degree >::SupportStart];
}
template< unsigned int FEMSig >
template< unsigned int D >
double BSplineEvaluationData< FEMSig >::CenterEvaluator< D >::ChildEvaluator::value( int fIdx , int cIdx , int d ) const
{
	int dd = cIdx-2*fIdx , res = 1<<(_parentDepth+1);
	if( cIdx<0 || cIdx>=res || OutOfBounds( _parentDepth , fIdx ) || dd<BSplineSupportSizes< Degree >::ChildSupportStart || dd>BSplineSupportSizes< Degree >::ChildSupportEnd ) return 0;
	return _pcValues[d][ OffsetToIndex( _parentDepth , fIdx ) ][dd-BSplineSupportSizes< Degree >::ChildSupportStart];
}
template< unsigned int FEMSig >
template< unsigned int D >
void BSplineEvaluationData< FEMSig >::SetCornerEvaluator( typename CornerEvaluator< D >::Evaluator& evaluator , int depth )
{
	evaluator._depth = depth;
	int res = 1<<depth;
	for( int i=0 ; i<IndexSize ; i++ ) for( int j=BSplineSupportSizes< Degree >::BCornerStart ; j<=BSplineSupportSizes< Degree >::BCornerEnd ; j++ )
	{
		int ii = IndexToOffset( depth , i );
		double s = ii + j;
		int jj = j-BSplineSupportSizes< Degree >::BCornerStart;
		for( int d1=0 ; d1<=D ; d1++ )
		{
			if( d1==Degree )
			{
				if     ( j==BSplineSupportSizes< Degree >::BCornerStart ) evaluator._ccValues[d1][i][jj] = (                                            Value( depth , ii , ( s+0.5 )/res , d1 ) ) / 2;
				else if( j==BSplineSupportSizes< Degree >::BCornerEnd   ) evaluator._ccValues[d1][i][jj] = ( Value( depth , ii , ( s-0.5 )/res , d1 )                                            ) / 2;
				else                                                      evaluator._ccValues[d1][i][jj] = ( Value( depth , ii , ( s-0.5 )/res , d1 ) + Value( depth , ii , ( s+0.5 )/res , d1 ) ) / 2;
			}
			else evaluator._ccValues[d1][i][jj] = Value( depth , ii , s /res , d1 );
		}
	}
}
template< unsigned int FEMSig >
template< unsigned int D  >
void BSplineEvaluationData< FEMSig >::SetChildCornerEvaluator( typename CornerEvaluator< D >::ChildEvaluator& evaluator , int parentDepth )
{
	evaluator._parentDepth = parentDepth;
	int res = 1<<(parentDepth+1);
	for( int i=0 ; i<IndexSize ; i++ ) for( int j=BSplineSupportSizes< Degree >::ChildBCornerStart ; j<=BSplineSupportSizes< Degree >::ChildBCornerEnd ; j++ )
	{
		int ii = IndexToOffset( parentDepth , i );
		double s = 2*ii + j;
		int jj = j-BSplineSupportSizes< Degree >::ChildBCornerStart;
		for( int d1=0 ; d1<=D ; d1++ )
		{
			if( d1==Degree )
			{
				if     ( j==BSplineSupportSizes< Degree >::ChildBCornerStart ) evaluator._pcValues[d1][i][jj] = (                                                  Value( parentDepth , ii , ( s+0.5 )/res , d1 ) ) / 2;
				else if( j==BSplineSupportSizes< Degree >::ChildBCornerEnd   ) evaluator._pcValues[d1][i][jj] = ( Value( parentDepth , ii , ( s-0.5 )/res , d1 )                                                  ) / 2;
				else                                                           evaluator._pcValues[d1][i][jj] = ( Value( parentDepth , ii , ( s-0.5 )/res , d1 ) + Value( parentDepth , ii , ( s+0.5 )/res , d1 ) ) / 2;
			}
			else evaluator._pcValues[d1][i][jj] = Value( parentDepth , ii , s /res , d1 );
		}
	}
}
template< unsigned int FEMSig >
template< unsigned int D >
double BSplineEvaluationData< FEMSig >::CornerEvaluator< D >::Evaluator::value( int fIdx , int cIdx , int d ) const
{
	int dd = cIdx-fIdx , res = ( 1<<_depth ) + 1;
	if( cIdx<0 || cIdx>=res || OutOfBounds( _depth , fIdx ) || dd<BSplineSupportSizes< Degree >::BCornerStart || dd>BSplineSupportSizes< Degree >::BCornerEnd ) return 0;
	return _ccValues[d][ OffsetToIndex( _depth , fIdx ) ][dd-BSplineSupportSizes< Degree >::BCornerStart];
}
template< unsigned int FEMSig >
template< unsigned int D >
double BSplineEvaluationData< FEMSig >::CornerEvaluator< D >::ChildEvaluator::value( int fIdx , int cIdx , int d ) const
{
	int dd = cIdx-2*fIdx , res = ( 1<<(_parentDepth+1) ) + 1;
	if( cIdx<0 || cIdx>=res || OutOfBounds( _parentDepth , fIdx ) || dd<BSplineSupportSizes< Degree >::ChildBCornerStart || dd>BSplineSupportSizes< Degree >::ChildBCornerEnd ) return 0;
	return _pcValues[d][ OffsetToIndex( _parentDepth , fIdx ) ][dd-BSplineSupportSizes< Degree >::ChildBCornerStart];
}
template< unsigned int FEMSig >
void BSplineEvaluationData< FEMSig >::SetUpSampleEvaluator( UpSampleEvaluator& evaluator , int lowDepth )
{
	evaluator._lowDepth = lowDepth;
	for( int i=0 ; i<IndexSize ; i++ )
	{
		int ii = IndexToOffset( lowDepth , i );
		BSplineUpSamplingCoefficients b( lowDepth , ii );
		for( int j=0 ; j<BSplineSupportSizes< Degree >::UpSampleSize ; j++ ) evaluator._pcValues[i][j] = b[j];
	}
}
template< unsigned int FEMSig >
double BSplineEvaluationData< FEMSig >::UpSampleEvaluator::value( int pIdx , int cIdx ) const
{
	int dd = cIdx-2*pIdx;
	if( OutOfBounds( _lowDepth+1 , cIdx ) || OutOfBounds( _lowDepth , pIdx ) || dd<BSplineSupportSizes< Degree >::UpSampleStart || dd>BSplineSupportSizes< Degree >::UpSampleEnd ) return 0;
	return _pcValues[ OffsetToIndex( _lowDepth , pIdx ) ][dd-BSplineSupportSizes< Degree >::UpSampleStart];
}

//////////////////////////////////////////////////////////
// BSplineEvaluationData::BSplineUpSamplingCoefficients //
//////////////////////////////////////////////////////////
template< unsigned int FEMSig >
BSplineEvaluationData< FEMSig >::BSplineUpSamplingCoefficients::BSplineUpSamplingCoefficients( int depth , int offset )
{
	static const BoundaryType BType = FEMSignature< FEMSig >::BType;
	// [ 1/8 1/2 3/4 1/2 1/8]
	// [ 1 , 1 ] ->  [ 3/4 , 1/2 , 1/8 ] + [ 1/8 , 1/2 , 3/4 ] = [ 7/8 , 1 , 7/8 ]
	int dim = BSplineSupportSizes< Degree >::Nodes(depth) , _dim = BSplineSupportSizes< Degree >::Nodes(depth+1);
	bool reflect;
	offset = BSplineData< FEMSig >::RemapOffset( depth , offset , reflect );
	int multiplier = ( BType==BOUNDARY_DIRICHLET && reflect ) ? -1 : 1;
	bool useReflected = ( BType!=BOUNDARY_FREE ) && ( BSplineSupportSizes< Degree >::Inset || ( offset % ( dim-1 ) ) );
	int b[ BSplineSupportSizes< Degree >::UpSampleSize ];
	Polynomial< Degree+1 >::BinomialCoefficients( b );

	// Clear the values
	memset( _coefficients , 0 , sizeof(int) * BSplineSupportSizes< Degree >::UpSampleSize );

	// Get the array of coefficients, relative to the origin
	int* coefficients = _coefficients - ( 2*offset + BSplineSupportSizes< Degree >::UpSampleStart );
	for( int i=BSplineSupportSizes< Degree >::UpSampleStart ; i<=BSplineSupportSizes< Degree >::UpSampleEnd ; i++ )
	{
		int _offset = 2*offset+i;
		_offset = BSplineData< FEMSig >::RemapOffset( depth+1 , _offset , reflect );
		if( useReflected || !reflect )
		{
			int _multiplier = multiplier * ( ( BType==BOUNDARY_DIRICHLET && reflect ) ? -1 : 1 );
			coefficients[ _offset ] += b[ i-BSplineSupportSizes< Degree >::UpSampleStart ] * _multiplier;
		}
		// If we are not inset and we are at the boundary, use the reflection as well
		if( BType!=BOUNDARY_FREE && !BSplineSupportSizes< Degree >::Inset && ( offset % (dim-1) ) && !( _offset % (_dim-1) ) )
		{
			_offset = BSplineData< FEMSig >::RemapOffset( depth+1 , _offset , reflect );
			int _multiplier = multiplier * ( ( BType==BOUNDARY_DIRICHLET && reflect ) ? -1 : 1 );
			if( BType==BOUNDARY_DIRICHLET ) _multiplier *= -1;
			coefficients[ _offset ] += b[ i-BSplineSupportSizes< Degree >::UpSampleStart ] * _multiplier;
		}
	}
}

////////////////////////////
// BSplineIntegrationData //
////////////////////////////
template< unsigned int FEMSig1 , unsigned int FEMSig2 >
template< unsigned int D1 , unsigned int D2 >
double BSplineIntegrationData< FEMSig1 , FEMSig2 >::Dot( int depth1 ,  int off1 , int depth2 , int off2 )
{
	if( D1>Degree1 ) ERROR_OUT( "Taking more derivatives than the degree: " , D1 , " > " , Degree1 );
	if( D2>Degree2 ) ERROR_OUT( "Taking more derivatives than the degree: " , D2 , " > " , Degree2 );
	const int _Degree1 = ( Degree1>=D1 ) ? Degree1 - D1 : 0 , _Degree2 = ( Degree2>=D2 ) ? Degree2 - D2 : 0;
	int sums[ Degree1+1 ][ Degree2+1 ];

	int depth = std::max< int >( depth1 , depth2 );

	BSplineElements< Degree1 > b1;
	BSplineElements< Degree2 > b2;
	if( BSplineSupportSizes< Degree1 >::IsInteriorlySupported( depth1 , off1 ) && BSplineSupportSizes< Degree2 >::IsInteriorlySupported( depth2 , off2 ) )
	{
		if( depth1<depth2 )
		{
			int begin1 , end1 , res = 1 - BSplineSupportSizes< Degree1 >::SupportStart + BSplineSupportSizes< Degree1 >::SupportEnd;
			BSplineSupportSizes< Degree1 >::InteriorSupportedSpan( depth1 , begin1 , end1 );
			b1 = BSplineElements< Degree1 >( res , begin1 , BOUNDARY_FREE );
			for( int d=depth1 ; d<depth2 ; d++ )
			{
				BSplineElements< Degree1 > b=b1;
				b.upSample( b1 );
				res <<= 1;
			}
			b2 = BSplineElements< Degree2 >( res , off2 - ( (off1-begin1)<<(depth2-depth1) ) , BOUNDARY_FREE );
		}
		else
		{
			int begin2 , end2 , res = 1 - BSplineSupportSizes< Degree2 >::SupportStart + BSplineSupportSizes< Degree2 >::SupportEnd;
			BSplineSupportSizes< Degree2 >::InteriorSupportedSpan( depth2 , begin2 , end2 );
			b2 = BSplineElements< Degree2 >( res , begin2 , BOUNDARY_FREE );
			for( int d=depth2 ; d<depth1 ; d++ )
			{
				BSplineElements< Degree2 > b=b2;
				b.upSample( b2 );
				res <<= 1;
			}
			b1 = BSplineElements< Degree1 >( res , off1 - ( (off2-begin2)<<(depth1-depth2) ) , BOUNDARY_FREE );
		}
	}
	else
	{
		b1 = BSplineElements< Degree1 >( 1<<depth1 , off1 , FEMSignature< FEMSig1 >::BType );
		b2 = BSplineElements< Degree2 >( 1<<depth2 , off2 , FEMSignature< FEMSig2 >::BType );
		{
			BSplineElements< Degree1 > b;
			while( depth1<depth ) b=b1 , b.upSample( b1 ) , depth1++;
		}
		{
			BSplineElements< Degree2 > b;
			while( depth2<depth ) b=b2 , b.upSample( b2 ) , depth2++;
		}
	}

	BSplineElements< Degree1-D1 > db1;
	BSplineElements< Degree2-D2 > db2;
	b1.template differentiate< D1 >( db1 ) , b2.template differentiate< D2 >( db2 );

	int start1=-1 , end1=-1 , start2=-1 , end2=-1;
	for( int i=0 ; i<int( b1.size() ) ; i++ )
	{
		for( int j=0 ; j<=Degree1 ; j++ )
		{
			if( b1[i][j] && start1==-1 ) start1 = i;
			if( b1[i][j] ) end1 = i+1;
		}
		for( int j=0 ; j<=Degree2 ; j++ )
		{
			if( b2[i][j] && start2==-1 ) start2 = i;
			if( b2[i][j] ) end2 = i+1;
		}
	}
	if( start1==end1 || start2==end2 || start1>=end2 || start2>=end1 ) return 0.;
	int start = std::max< int >( start1 , start2 ) , end = std::min< int >( end1 , end2 );
	memset( sums , 0 , sizeof( sums ) );

	// Iterate over the support
	for( int i=start ; i<end ; i++ )
		// Iterate over all pairs of elements within a node
		for( int j=0 ; j<=_Degree1 ; j++ ) for( int k=0 ; k<=_Degree2 ; k++ )
			// Accumulate the product of the coefficients
			sums[j][k] += db1[i][j] * db2[i][k];

	double _dot = 0;
	{
		double integrals[ _Degree1+1 ][ _Degree2+1 ];
		SetBSplineElementIntegrals< _Degree1 , _Degree2 >( integrals );
		for( int j=0 ; j<=_Degree1 ; j++ ) for( int k=0 ; k<=_Degree2 ; k++ ) _dot += integrals[j][k] * sums[j][k];
	}
	_dot /= b1.denominator;
	_dot /= b2.denominator;
	return ( !D1 && !D2 ) ? _dot / (1<<depth) : _dot * ( 1<<( depth*(D1+D2-1) ) );
}
template< unsigned int FEMSig1 , unsigned int FEMSig2 >
template< unsigned int D1 , unsigned int D2 , unsigned int _D1 , unsigned int _D2 , class Integrator >
void BSplineIntegrationData< FEMSig1 , FEMSig2 >::IntegratorSetter< D1 , D2 , _D1 , _D2 , Integrator >::Set2D( Integrator& integrator , int depth )
{
	IntegratorSetter< D1-1 , D2 , _D1 , _D2 , Integrator >::Set2D( integrator , depth );
	IntegratorSetter< D1   , D2 , _D1 , _D2 , Integrator >::Set1D( integrator , depth );
}
template< unsigned int FEMSig1 , unsigned int FEMSig2 >
template< unsigned int D1 , unsigned int D2 , unsigned int _D1 , unsigned int _D2 , class Integrator >
void BSplineIntegrationData< FEMSig1 , FEMSig2 >::IntegratorSetter< D1 , D2 , _D1 , _D2 , Integrator >::Set1D( Integrator& integrator , int depth )
{
	IntegratorSetter< D1 , D2-1 , _D1 , _D2 , Integrator >::Set1D( integrator , depth );
	_IntegratorSetter< D1 , D2 , _D1 , _D2 >::Set( integrator , depth );
}
template< unsigned int FEMSig1 , unsigned int FEMSig2 >
template< unsigned int D2 , unsigned int _D1 , unsigned int _D2 , class Integrator >
void BSplineIntegrationData< FEMSig1 , FEMSig2 >::IntegratorSetter< 0 , D2 , _D1 , _D2 , Integrator >::Set2D( Integrator& integrator , int depth )
{
	IntegratorSetter< 0 , D2 , _D1 , _D2 , Integrator >::Set1D( integrator , depth );
}
template< unsigned int FEMSig1 , unsigned int FEMSig2 >
template< unsigned int D2 , unsigned int _D1 , unsigned int _D2 , class Integrator >
void BSplineIntegrationData< FEMSig1 , FEMSig2 >::IntegratorSetter< 0 , D2 , _D1 , _D2 , Integrator >::Set1D( Integrator& integrator , int depth )
{
	IntegratorSetter< 0 , D2-1 , _D1 , _D2 , Integrator >::Set1D( integrator , depth );
	_IntegratorSetter< 0 , D2 , _D1 , _D2 >::Set( integrator , depth );
}
template< unsigned int FEMSig1 , unsigned int FEMSig2 >
template< unsigned int D1 , unsigned int _D1 , unsigned int _D2 , class Integrator >
void BSplineIntegrationData< FEMSig1 , FEMSig2 >::IntegratorSetter< D1 , 0 , _D1 , _D2 , Integrator >::Set2D( Integrator& integrator , int depth )
{
	IntegratorSetter< D1-1 , 0 , _D1 , _D2 , Integrator >::Set2D( integrator , depth );
	IntegratorSetter< D1   , 0 , _D1 , _D2 , Integrator >::Set1D( integrator , depth );
}
template< unsigned int FEMSig1 , unsigned int FEMSig2 >
template< unsigned int D1 , unsigned int _D1 , unsigned int _D2 , class Integrator >
void BSplineIntegrationData< FEMSig1 , FEMSig2 >::IntegratorSetter< D1 , 0 , _D1 , _D2 , Integrator >::Set1D( Integrator& integrator , int depth )
{
	_IntegratorSetter< D1 , 0 , _D1 , _D2 >::Set( integrator , depth );
}
template< unsigned int FEMSig1 , unsigned int FEMSig2 >
template< unsigned int _D1 , unsigned int _D2 , class Integrator >
void BSplineIntegrationData< FEMSig1 , FEMSig2 >::IntegratorSetter< 0 , 0 , _D1 , _D2 , Integrator >::Set2D( Integrator& integrator , int depth )
{
	IntegratorSetter< 0 , 0 , _D1 , _D2 , Integrator >::Set1D( integrator , depth );
}
template< unsigned int FEMSig1 , unsigned int FEMSig2 >
template< unsigned int _D1 , unsigned int _D2 , class Integrator >
void BSplineIntegrationData< FEMSig1 , FEMSig2 >::IntegratorSetter< 0 , 0 , _D1 , _D2 , Integrator >::Set1D( Integrator& integrator , int depth )
{
	_IntegratorSetter< 0 , 0 , _D1 , _D2 >::Set( integrator , depth );
}

template< unsigned int FEMSig1 , unsigned int FEMSig2 >
template< unsigned int D1 , unsigned int D2 , unsigned int _D1 , unsigned int _D2 >
void BSplineIntegrationData< FEMSig1 , FEMSig2 >::_IntegratorSetter< D1 , D2 , _D1 , _D2 >::Set( typename FunctionIntegrator::template Integrator< _D1 , _D2 >& integrator , int depth )
{
	for( int i=0 ; i<IndexSize ; i++ ) for( int j=BSplineOverlapSizes< Degree1 , Degree2 >::OverlapStart ; j<=BSplineOverlapSizes< Degree1 , Degree2 >::OverlapEnd ; j++ )
	{
		int ii = IndexToOffset( depth , i );
		integrator._ccIntegrals[D1][D2][i][j-BSplineOverlapSizes< Degree1 , Degree2 >::OverlapStart] = Dot< D1 , D2 >( depth , ii , depth , ii+j );
	}
}
template< unsigned int FEMSig1 , unsigned int FEMSig2 >
template< unsigned int D1 , unsigned int D2 , unsigned int _D1 , unsigned int _D2 >
void BSplineIntegrationData< FEMSig1 , FEMSig2 >::_IntegratorSetter< D1 , D2 , _D1 , _D2 >::Set( typename FunctionIntegrator::template ChildIntegrator< _D1 , _D2 >& integrator , int pDepth )
{
	for( int i=0 ; i<IndexSize ; i++ ) for( int j=BSplineOverlapSizes< Degree1 , Degree2 >::ChildOverlapStart ; j<=BSplineOverlapSizes< Degree1 , Degree2 >::ChildOverlapEnd ; j++ )
	{
		int ii = IndexToOffset( pDepth , i );
		integrator._pcIntegrals[D1][D2][i][j-BSplineOverlapSizes< Degree1 , Degree2 >::ChildOverlapStart] = Dot< D1 , D2 >( pDepth , ii , pDepth+1 , 2*ii+j );
	}
}

template< unsigned int FEMSig1 , unsigned int FEMSig2 >
template< unsigned int D1 , unsigned int D2 >
void BSplineIntegrationData< FEMSig1 , FEMSig2 >::SetIntegrator( typename FunctionIntegrator::template Integrator< D1 , D2 >& integrator , int depth )
{
	integrator._depth = depth;
	IntegratorSetter< D1 , D2 , D1 , D2 , typename FunctionIntegrator::template Integrator< D1 , D2 > >::Set2D( integrator , depth );
}
template< unsigned int FEMSig1 , unsigned int FEMSig2 >
template< unsigned int D1 , unsigned int D2 >
void BSplineIntegrationData< FEMSig1 , FEMSig2 >::SetChildIntegrator( typename FunctionIntegrator::template ChildIntegrator< D1 , D2 >& integrator , int parentDepth )
{
	integrator._parentDepth = parentDepth;
	IntegratorSetter< D1 , D2 , D1 , D2 , typename FunctionIntegrator::template ChildIntegrator< D1 , D2 > >::Set2D( integrator , parentDepth );
}
template< unsigned int FEMSig1 , unsigned int FEMSig2 >
template< unsigned int D1 , unsigned int D2 >
double BSplineIntegrationData< FEMSig1 , FEMSig2 >::FunctionIntegrator::Integrator< D1 , D2 >::dot( int off1 , int off2 , int d1 , int d2 ) const
{
	int d = off2-off1;
	if( BSplineEvaluationData< FEMSig1 >::OutOfBounds( _depth , off1 ) || BSplineEvaluationData< FEMSig2 >::OutOfBounds( _depth , off2 ) || d<BSplineOverlapSizes< Degree1 , Degree2 >::OverlapStart || d>BSplineOverlapSizes< Degree1 , Degree2 >::OverlapEnd ) return 0;
	return _ccIntegrals[d1][d2][ OffsetToIndex( _depth , off1 ) ][d-BSplineOverlapSizes< Degree1 , Degree2 >::OverlapStart];
}
template< unsigned int FEMSig1 , unsigned int FEMSig2 >
template< unsigned int D1 , unsigned int D2 >
double BSplineIntegrationData< FEMSig1 , FEMSig2 >::FunctionIntegrator::ChildIntegrator< D1 , D2 >::dot( int off1 , int off2 , int d1 , int d2 ) const
{
	int d = off2-2*off1;
	if( BSplineEvaluationData< FEMSig1 >::OutOfBounds( _parentDepth , off1 ) || BSplineEvaluationData< FEMSig2 >::OutOfBounds( _parentDepth+1 , off2 ) || d<BSplineOverlapSizes< Degree1 , Degree2 >::ChildOverlapStart || d>BSplineOverlapSizes< Degree1 , Degree2 >::ChildOverlapEnd ) return 0;
	return _pcIntegrals[d1][d2][ OffsetToIndex( _parentDepth , off1 ) ][d-BSplineOverlapSizes< Degree1 , Degree2 >::ChildOverlapStart];
}

/////////////////
// BSplineData //
/////////////////
#define MODULO( A , B ) ( (A)<0 ? ( (B)-((-(A))%(B)) ) % (B) : (A) % (B) )

template< unsigned int FEMSig , unsigned int D >
BSplineData< FEMSig , D >::BSplineComponents::BSplineComponents( int depth , int offset )
{
	static const int _Degree = Degree;
	int res = 1<<depth;
	BSplineElements< Degree > elements( res , offset , FEMSignature< FEMSig >::BType );

	// The first index is the position, the second is the element type
	Polynomial< Degree > components[Degree+1][Degree+1];
	// Generate the elements that can appear in the base function corresponding to the base function at (depth,offset) = (0,0)
	for( int d=0 ; d<=Degree ; d++ ) for( int dd=0 ; dd<=Degree ; dd++ ) components[d][dd] = Polynomial< Degree >::BSplineComponent( _Degree-dd ).shift( -( (_Degree+1)/2 ) + d );

	// Now adjust to the desired depth and offset
	double width = 1. / res;
	for( int d=0 ; d<=Degree ; d++ ) for( int dd=0 ; dd<=Degree ; dd++ ) components[d][dd] = components[d][dd].scale( width ).shift( width*offset );

	// Now write in the polynomials
	for( int d=0 ; d<=Degree ; d++ )
	{
		int idx = offset + BSplineSupportSizes< Degree >::SupportStart + d;
		_polys[d][0] = Polynomial< Degree >();

		if( idx>=0 && idx<res ) for( int dd=0 ; dd<=Degree ; dd++ ) _polys[d][0] += components[d][dd] * ( ( double )( elements[idx][dd] ) ) / elements.denominator;
	}
	for( int d=1 ; d<=D ; d++ ) for( int dd=0 ; dd<=Degree ; dd++ ) _polys[dd][d] = _polys[dd][d-1].derivative();
}

template< unsigned int FEMSig , unsigned int D >
int BSplineData< FEMSig , D >::RemapOffset( int depth , int offset , bool& reflect )
{
	const int I = ( Degree&1 ) ? 0 : 1;
	if( FEMSignature< FEMSig >::BType==BOUNDARY_FREE ){ reflect = false ; return offset; }
	int dim = BSplineEvaluationData< FEMDegreeAndBType< Degree , BOUNDARY_NEUMANN >::Signature >::End( depth ) - BSplineEvaluationData< FEMDegreeAndBType< Degree , BOUNDARY_NEUMANN >::Signature >::Begin( depth );
	offset = MODULO( offset , 2*(dim-1+I) );
	reflect = offset>=dim;
	if( reflect ) return 2*(dim-1+I) - (offset+I);
	else          return offset;
}
#undef MODULO

template< unsigned int FEMSig , unsigned int D >
BSplineData< FEMSig , D >::BSplineData( void )
{
	_maxDepth = 0;
	_evaluators = NullPointer( SparseBSplineEvaluator );
}
template< unsigned int FEMSig , unsigned int D >
void BSplineData< FEMSig , D >::reset( int maxDepth )
{
	if( _evaluators ) DeletePointer( _evaluators );

	_maxDepth = maxDepth;
	_evaluators = NewPointer< SparseBSplineEvaluator >( _maxDepth+1 );
	for( unsigned int d=0 ; d<=_maxDepth ; d++ ) _evaluators[d].init( d );
}
template< unsigned int FEMSig , unsigned int D >
BSplineData< FEMSig , D >::BSplineData( int maxDepth )
{
	_evaluators = NullPointer( SparseBSplineEvaluator );
	reset( maxDepth );
}
template< unsigned int FEMSig , unsigned int D >
BSplineData< FEMSig , D >::~BSplineData( void )
{
	DeletePointer( _evaluators );
}

/////////////////////
// BSplineElements //
/////////////////////
template< unsigned int Degree >
BSplineElements< Degree >::BSplineElements( int res , int offset , BoundaryType bType )
{
	denominator = 1;
	std::vector< BSplineElementCoefficients< Degree > >::resize( res , BSplineElementCoefficients< Degree >() );

	// If we have primal dirichlet constraints, the boundary functions are necessarily zero
	if( _Primal && bType==BOUNDARY_DIRICHLET && !(offset%res) ) return;

	// Construct the B-Spline
	for( int i=0 ; i<=Degree ; i++ )
	{
		int idx = -_Off + offset + i;
		if( idx>=0 && idx<res ) (*this)[idx][i] = 1;
	}
	if( bType!=BOUNDARY_FREE )
	{
		// Fold in the periodic instances (which cancels the negation)
		_addPeriodic< true >( _RotateLeft ( offset , res ) , false ) , _addPeriodic< false >( _RotateRight( offset , res ) , false );

		// Recursively fold in the boundaries
		if( _Primal && !(offset%res) ) return;

		// Fold in the reflected instance (which may require negation)
		_addPeriodic< true >( _ReflectLeft( offset , res ) , bType==BOUNDARY_DIRICHLET ) , _addPeriodic< false >( _ReflectRight( offset , res ) , bType==BOUNDARY_DIRICHLET );
	}
}
template< unsigned int Degree > int BSplineElements< Degree >::_ReflectLeft ( int offset , int res ){ return (Degree&1) ?      -offset :      -1-offset; }
template< unsigned int Degree > int BSplineElements< Degree >::_ReflectRight( int offset , int res ){ return (Degree&1) ? 2*res-offset : 2*res-1-offset; }
template< unsigned int Degree > int BSplineElements< Degree >::_RotateLeft  ( int offset , int res ){ return offset-2*res; }
template< unsigned int Degree > int BSplineElements< Degree >::_RotateRight ( int offset , int res ){ return offset+2*res; }

template< unsigned int Degree >
template< bool Left >
void BSplineElements< Degree >::_addPeriodic( int offset , bool negate )
{
	int res = int( std::vector< BSplineElementCoefficients< Degree > >::size() );
	bool set = false;
	// Add in the corresponding B-spline elements (possibly negated)
	for( int i=0 ; i<=Degree ; i++ )
	{
		int idx = -_Off + offset + i;
		if( idx>=0 && idx<res ) (*this)[idx][i] += negate ? -1 : 1 , set = true;
	}
	// If there is a change for additional overlap, give it a go
	if( set ) _addPeriodic< Left >( Left ? _RotateLeft( offset , res ) : _RotateRight( offset , res ) , negate );
}
template< unsigned int Degree >
void BSplineElements< Degree >::upSample( BSplineElements< Degree >& high ) const
{
	int bCoefficients[ BSplineSupportSizes< Degree >::UpSampleSize ];
	Polynomial< Degree+1 >::BinomialCoefficients( bCoefficients );

	high.resize( std::vector< BSplineElementCoefficients< Degree > >::size()*2 );
	high.assign( high.size() , BSplineElementCoefficients< Degree >() );
	// [NOTE] We have flipped the order of the B-spline elements
	for( int i=0 ; i<int(std::vector< BSplineElementCoefficients< Degree > >::size()) ; i++ ) for( int j=0 ; j<=Degree ; j++ )
	{
		// At index I , B-spline element J corresponds to a B-spline centered at:
		//		I - SupportStart - J
		int idx = i - BSplineSupportSizes< Degree >::SupportStart - j;
		for( int k=BSplineSupportSizes< Degree >::UpSampleStart ; k<=BSplineSupportSizes< Degree >::UpSampleEnd ; k++ )
		{
			// Index idx at the coarser resolution gets up-sampled into indices:
			//		2*idx + [UpSampleStart,UpSampleEnd]
			// at the finer resolution
			int _idx = 2*idx + k;
			// Compute the index of the B-spline element relative to 2*i and 2*i+1
			int _j1 = -_idx + 2*i - BSplineSupportSizes< Degree >::SupportStart , _j2 = -_idx + 2*i + 1 - BSplineSupportSizes< Degree >::SupportStart;
			if( _j1>=0 && _j1<=Degree ) high[2*i+0][_j1] += (*this)[i][j] * bCoefficients[k-BSplineSupportSizes< Degree >::UpSampleStart];
			if( _j2>=0 && _j2<=Degree ) high[2*i+1][_j2] += (*this)[i][j] * bCoefficients[k-BSplineSupportSizes< Degree >::UpSampleStart];
		}
	}
	high.denominator = denominator<<Degree;
}

template< unsigned int Degree >
template< unsigned int D >
void BSplineElements< Degree >::differentiate( BSplineElements< Degree-D >& d ) const{ Differentiator< Degree , Degree-D >::Differentiate( *this , d ); }
template< unsigned int Degree , unsigned int DDegree >
void Differentiator< Degree , DDegree >::Differentiate( const BSplineElements< Degree >& bse , BSplineElements< DDegree >& dbse )
{
	BSplineElements< Degree-1 > _dbse;
	_dbse.resize( bse.size() );
	_dbse.assign( _dbse.size() , BSplineElementCoefficients< Degree-1 >() );
	for( int i=0 ; i<(int)bse.size() ; i++ ) for( int j=0 ; j<=Degree ; j++ )
	{
		if( j-1>=0 )   _dbse[i][j-1] -= bse[i][j];
		if( j<Degree ) _dbse[i][j  ] += bse[i][j];
	}
	_dbse.denominator = bse.denominator;
	return Differentiator< Degree-1 , DDegree >::Differentiate( _dbse , dbse );
}
template< unsigned int Degree >
void Differentiator< Degree , Degree >::Differentiate( const BSplineElements< Degree >& bse , BSplineElements< Degree >& dbse ){ dbse = bse; }

// If we were really good, we would implement this integral table to store
// rational values to improve precision...
template< unsigned int Degree1 , unsigned int Degree2 >
void SetBSplineElementIntegrals( double integrals[Degree1+1][Degree2+1] )
{
	for( int i=0 ; i<=Degree1 ; i++ )
	{
		Polynomial< Degree1 > p1 = Polynomial< Degree1 >::BSplineComponent( Degree1-i );
		for( int j=0 ; j<=Degree2 ; j++ )
		{
			Polynomial< Degree2 > p2 = Polynomial< Degree2 >::BSplineComponent( Degree2-j );
			integrals[i][j] = ( p1 * p2 ).integral( 0 , 1 );
		}
	}
}
