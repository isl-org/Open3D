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

template< class T , class const_iterator > size_t SparseMatrixInterface< T , const_iterator >::entries( void ) const
{
	size_t entries = 0;
	for( size_t i=0 ; i<rows() ; i++ ) entries += rowSize( i );
	return entries;
}
template< class T , class const_iterator > double SparseMatrixInterface< T , const_iterator >::squareNorm( void ) const
{
	double n=0;
	for( size_t i=0 ; i<rows() ; i++ )
	{
		const_iterator e = end( i );
		for( const_iterator iter = begin( i ) ; iter!=e ; iter++ ) n += iter->Value * iter->Value;
	}
	return n;

}
template< class T , class const_iterator > double SparseMatrixInterface< T , const_iterator >::squareASymmetricNorm( void ) const
{
	double n=0;
	for( size_t i=0 ; i<rows() ; i++ )
	{
		const_iterator e = end( i );
		for( const_iterator iter1 = begin( i ) ; iter1!=e ; iter1++ )
		{
			size_t j = iter1->N;
			const_iterator e = end( j );
			double value = 0;
			for( const_iterator iter2 = begin( j ) ; iter2!=e ; iter2++ )
			{
				size_t k = iter2->N;
				if( k==i ) value += iter2->Value;
			}
			n += (iter1->Value-value) * (iter1->Value-value);
		}
	}
	return n;
}
template< class T , class const_iterator > double SparseMatrixInterface< T , const_iterator >::squareASymmetricNorm( size_t &idx1 , size_t &idx2 ) const
{
	double n=0;
	double max=0;
	for( size_t i=0 ; i<rows() ; i++ )
	{
		const_iterator e = end( i );
		for( const_iterator iter = begin( i ) ; iter!=e ; iter++ )
		{
			size_t j = iter->N;
			const_iterator e = end( j );
			double value = 0;
			for( const_iterator iter2 = begin( j ) ; iter2!=e ; iter2++ )
			{
				size_t k = iter2->N;
				if( k==i ) value += iter2->Value;
			}
			double temp = (iter->Value-value) * (iter->Value-value);
			n += temp;
			if( temp>=max ) idx1 = i , idx2 = j , max=temp;
		}
	}
	return n;
}
template< class T , class const_iterator >
template< class T2 >
void SparseMatrixInterface< T , const_iterator >::multiply( ConstPointer( T2 ) In , Pointer( T2 ) Out , char multiplyFlag ) const
{
	ConstPointer( T2 ) in = In;
	ThreadPool::Parallel_for( 0 , rows() , [&]( unsigned int , size_t i )
	{
		T2 temp;
		memset( &temp , 0 , sizeof(T2) );
		ConstPointer( T2 ) _in = in;
		const_iterator e = end( i );
		for( const_iterator iter = begin( i ) ; iter!=e ; iter++ ) temp += (T2)( _in[ iter->N ] * iter->Value );
		if( multiplyFlag & MULTIPLY_NEGATE ) temp = -temp;
		if( multiplyFlag & MULTIPLY_ADD ) Out[i] += temp;
		else                              Out[i]  = temp;
	}
	);
}
template< class T , class const_iterator >
template< class T2 >
void SparseMatrixInterface< T , const_iterator >::multiplyScaled( T scale , ConstPointer( T2 ) In , Pointer( T2 ) Out , char multiplyFlag ) const
{
	ConstPointer( T2 ) in = In;
	ThreadPool::Parallel_for( 0 , rows() , [&]( unsigned int , size_t i )
	{
		T2 temp;
		memset( &temp , 0 , sizeof(T2) );
		ConstPointer( T2 ) _in = in;
		const_iterator e = end( i );
		for( const_iterator iter = begin( i ) ; iter!=e ; iter++ ) temp += _in[ iter->N ] * iter->Value;
		temp *= scale;
		if( multiplyFlag & MULTIPLY_NEGATE ) temp = -temp;
		if( multiplyFlag & MULTIPLY_ADD ) Out[i] += temp;
		else                              Out[i]  = temp;
	}
	);
}

template< class T , class const_iterator >
void SparseMatrixInterface< T , const_iterator >::setDiagonal( Pointer( T ) diagonal ) const
{
	ThreadPool::Parallel_for( 0 , rows() , [&]( unsigned int , size_t i )
	{
		diagonal[i] = (T)0;
		const_iterator e = end( i );
		for( const_iterator iter = begin( i ) ; iter!=e ; iter++ ) if( iter->N==i ) diagonal[i] += iter->Value;
	}
	);
}

template< class T , class const_iterator >
void SparseMatrixInterface< T , const_iterator >::setDiagonalR( Pointer( T ) diagonal ) const
{
	ThreadPool::Parallel_for( 0 , rows() , [&]( unsigned int , size_t i )
	{
		diagonal[i] = (T)0;
		const_iterator e = end( i );
		for( const_iterator iter = begin( i ) ; iter!=e ; iter++ ) if( iter->N==i ) diagonal[i] += iter->Value;
		if( diagonal[i] ) diagonal[i] = (T)( 1./diagonal[i] );
	}
	);
}

template< class T , class const_iterator >
template< class T2 >
void SparseMatrixInterface< T , const_iterator >::jacobiIteration( ConstPointer( T ) diagonal , ConstPointer( T2 ) b , ConstPointer( T2 ) in , Pointer( T2 ) out , bool dReciprocal ) const
{
	multiply( in , out );
	if( dReciprocal )
		ThreadPool::Parallel_for( 0 , rows() , [&]( unsigned int , size_t i ){ out[i] = in[i] + ( b[i] - out[i] ) * diagonal[i]; } );
	else
		ThreadPool::Parallel_for( 0 , rows() , [&]( unsigned int , size_t i ){ out[i] = in[i] + ( b[i] - out[i] ) / diagonal[i]; } );
}
template< class T , class const_iterator >
template< class T2 >
void SparseMatrixInterface< T , const_iterator >::gsIteration( ConstPointer( T ) diagonal , ConstPointer( T2 ) b , Pointer( T2 ) x , bool forward , bool dReciprocal ) const
{
	if( dReciprocal )
	{
#define ITERATE( j )                                                                                \
	{                                                                                               \
		T2 _b = b[j];                                                                               \
		const_iterator e = end( j );                                                                \
		for( const_iterator iter = begin( j ) ; iter!=e ; iter++ ) _b -= x[iter->N] * iter->Value;  \
		x[j] += _b * diagonal[j];                                                                   \
	}
		if( forward ) for( long long j=0 ; j<(long long)rows() ; j++ ){ ITERATE( j ); }
		else          for( long long j=(long long)rows()-1 ; j>=0 ; j-- ){ ITERATE( j ); }
#undef ITERATE
	}
	else
	{
#define ITERATE( j )                                                                                \
	{                                                                                               \
		T2 _b = b[j];                                                                               \
		const_iterator e = end( j );                                                                \
		for( const_iterator iter = begin( j ) ; iter!=e ; iter++ ) _b -= x[iter->N] * iter->Value;  \
		x[j] += _b / diagonal[j];                                                                   \
	}

		if( forward ) for( long long j=0 ; j<(long long)rows() ; j++ ){ ITERATE( j ); }
		else          for( long long j=(long long)rows()-1 ; j>=0 ; j-- ){ ITERATE( j ); }
#undef ITERATE
	}
}
template< class T , class const_iterator >
template< class T2 >
void SparseMatrixInterface< T , const_iterator >::gsIteration( const std::vector< size_t >& multiColorIndices , ConstPointer( T ) diagonal , ConstPointer( T2 ) b , Pointer( T2 ) x , bool dReciprocal ) const
{
	if( dReciprocal )
		ThreadPool::Parallel_for( 0 , multiColorIndices.size() , [&]( unsigned int , size_t j )
		{
			size_t jj = multiColorIndices[j];
			T2 _b = b[jj];
			const_iterator e = end( jj );
			for( const_iterator iter = begin( jj ) ; iter!=e ; iter++ ) _b -= x[iter->N] * iter->Value;
			x[jj] += _b * diagonal[jj];
		}
	);
	else
			ThreadPool::Parallel_for( 0 , multiColorIndices.size() , [&]( unsigned int , size_t j )
		{
			size_t jj = multiColorIndices[j];
			T2 _b = b[jj];
			const_iterator e = end( jj );
			for( const_iterator iter = begin( jj ) ; iter!=e ; iter++ ) _b -= x[iter->N] * iter->Value;
			x[jj] += _b / diagonal[jj];
		}
	);
}

template< class T , class const_iterator >
template< class T2 >
void SparseMatrixInterface< T , const_iterator >::gsIteration( const std::vector< std::vector< size_t > >& multiColorIndices , ConstPointer( T ) diagonal , ConstPointer( T2 ) b , Pointer( T2 ) x , bool forward , bool dReciprocal ) const
{
	if( dReciprocal )
	{
#define ITERATE( indices )                                                                               \
	{                                                                                                    \
		ThreadPool::Parallel_for( 0 , indices.size() , [&]( unsigned int , size_t k )                             \
		{                                                                                                \
			size_t jj = indices[k];                                                                      \
			T2 _b = b[jj];                                                                               \
			const_iterator e = end( jj );                                                                \
			for( const_iterator iter = begin( jj ) ; iter!=e ; iter++ ) _b -= x[iter->N] * iter->Value;  \
			x[jj] += _b * diagonal[jj];                                                                  \
		}                                                                                                \
		);                                                                                               \
	}
		if( forward ) for( size_t j=0 ; j<multiColorIndices.size() ; j++ ){ ITERATE( multiColorIndices[j] ); }
		else for( long long j=(long long)multiColorIndices.size()-1 ; j>=0 ; j-- ){ ITERATE( multiColorIndices[j] ); }
#undef ITERATE
	}
	else
	{
#define ITERATE( indices )                                                                               \
	{                                                                                                    \
		ThreadPool::Parallel_for( 0 , indices.size() , [&]( unsigned int , size_t k )                             \
		{                                                                                                \
			size_t jj = indices[k];                                                                      \
			T2 _b = b[jj];                                                                               \
			const_iterator e = end( jj );                                                                \
			for( const_iterator iter = begin( jj ) ; iter!=e ; iter++ ) _b -= x[iter->N] * iter->Value;  \
			x[jj] += _b / diagonal[jj];                                                                  \
		}                                                                                                \
		);                                                                                               \
	}
		if( forward ) for( size_t j=0 ; j<multiColorIndices.size()  ; j++ ){ ITERATE( multiColorIndices[j] ); }
		else for( long long j=(long long)multiColorIndices.size()-1 ; j>=0 ; j-- ){ ITERATE( multiColorIndices[j] ); }
#undef ITERATE
	}
}
template< class SPDFunctor , class T , typename Real , class TDotTFunctor > size_t SolveCG( const SPDFunctor& M , size_t dim , ConstPointer( T ) b , size_t iters , Pointer( T ) x , double eps , TDotTFunctor Dot )
{
	std::vector< Real > scratch( ThreadPool::NumThreads() , 0 );
	eps *= eps;
	Pointer( T ) r = AllocPointer< T >( dim );
	Pointer( T ) d = AllocPointer< T >( dim );
	Pointer( T ) q = AllocPointer< T >( dim );

	Real delta_new = 0 , delta_0;
	M( ( ConstPointer( T ) )x , r );
	ThreadPool::Parallel_for( 0 , dim , [&]( unsigned int thread , size_t i ){ d[i] = r[i] = b[i] - r[i] , scratch[thread] += Dot( r[i] , r[i] ); } );
	for( unsigned int t=0 ; t<ThreadPool::NumThreads() ; t++ ){ delta_new += scratch[t] ; scratch[t] = 0; }

	delta_0 = delta_new;
	if( delta_new<=eps )
	{
		FreePointer( r );
		FreePointer( d );
		FreePointer( q );
		return 0;
	}
	size_t ii;
	for( ii=0 ; ii<iters && delta_new>eps*delta_0 ; ii++ )
	{
		M( ( ConstPointer( T ) )d , q );
		Real dDotQ = 0;
		ThreadPool::Parallel_for( 0 , dim , [&]( unsigned int thread , size_t i ){ scratch[thread] += Dot( d[i] , q[i] ); } );
		for( unsigned int t=0 ; t<ThreadPool::NumThreads() ; t++ ){ dDotQ += scratch[t] ; scratch[t] = 0; }
		if( !dDotQ ) break;

		Real alpha = delta_new / dDotQ;
		Real delta_old = delta_new;
		delta_new = 0;
		if( (ii%50)==(50-1) )
		{
			ThreadPool::Parallel_for( 0 , dim , [&]( unsigned int , size_t i ){ x[i] += (T)( d[i] * alpha ); } );
			M( ( ConstPointer( T ) )x , r );
			ThreadPool::Parallel_for( 0 , dim , [&]( unsigned int thread , size_t i ){ r[i] = b[i] - r[i] , scratch[thread] += Dot( r[i] , r[i] ) , x[i] += (T)( d[i] * alpha ); } );
			for( unsigned int t=0 ; t<ThreadPool::NumThreads() ; t++ ){ delta_new += scratch[t] ; scratch[t] = 0; }
		}
		else
		{
			ThreadPool::Parallel_for( 0 , dim , [&]( unsigned int thread , size_t i ){ r[i] -=(T)( q[i] * alpha ) , scratch[thread] += Dot( r[i] , r[i] ) ,  x[i] += (T)( d[i] * alpha ); } );
			for( unsigned int t=0 ; t<ThreadPool::NumThreads() ; t++ ){ delta_new += scratch[t] ; scratch[t] = 0; }
		}

		Real beta = delta_new / delta_old;
		ThreadPool::Parallel_for( 0 , dim , [&]( unsigned int , size_t i ){ d[i] = r[i] + (T)( d[i] * beta ); } );
	}
	FreePointer( r );
	FreePointer( d );
	FreePointer( q );
	return ii;
}
template< class SPDFunctor , class Preconditioner , class T , typename Real , class TDotTFunctor > size_t SolveCG( const SPDFunctor& M , const Preconditioner& P , size_t dim , ConstPointer( T ) b , size_t iters , Pointer( T ) x , double eps , TDotTFunctor Dot  )
{
	std::vector< Real > scratch( ThreadPool::NumThreads() , 0 );
	eps *= eps;
	Pointer( T ) r = AllocPointer< T >( dim );
	Pointer( T ) d = AllocPointer< T >( dim );
	Pointer( T ) q = AllocPointer< T >( dim );
	Pointer( T ) Pb = AllocPointer< T >( dim );
	Pointer( T ) temp = AllocPointer< T >( dim );

	auto PM = [&] ( ConstPointer(T) x , Pointer(T) y )
	{
		M( x , temp );
		P( ( ConstPointer(T) )temp , y );
	};

	Real delta_new = 0 , delta_0;
	P( b , Pb );
	PM( ( ConstPointer( T ) )x , r );
	ThreadPool::Parallel_for( 0 , dim , [&]( unsigned int thread , size_t i ){ d[i] = r[i] = Pb[i] - r[i] , scratch[thread] += Dot( r[i] , r[i] ); } );
	for( unsigned int t=0 ; t<ThreadPool::NumThreads() ; t++ ){ delta_new += scratch[t] ; scratch[t] = 0; }

	delta_0 = delta_new;
	if( delta_new<=eps )
	{
		FreePointer( Pb );
		FreePointer( r );
		FreePointer( d );
		FreePointer( q );
		FreePointer( temp );
		return 0;
	}
	size_t ii;
	for( ii=0 ; ii<iters && delta_new>eps*delta_0 ; ii++ )
	{
		PM( ( ConstPointer( T ) )d , q );
		Real dDotQ = 0;
		ThreadPool::Parallel_for( 0 , dim , [&]( unsigned int thread , size_t i ){ scratch[thread] += Dot( d[i] , q[i] ); } );
		for( unsigned int t=0 ; t<ThreadPool::NumThreads() ; t++ ){ dDotQ += scratch[t] ; scratch[t] = 0; }
		if( !dDotQ ) break;

		Real alpha = delta_new / dDotQ;
		Real delta_old = delta_new;
		delta_new = 0;
		if( (ii%50)==(50-1) )
		{
			ThreadPool::Parallel_for( 0 , dim , [&]( unsigned int , size_t i ){ x[i] += (T)( d[i] * alpha ); } );
			PM( ( ConstPointer( T ) )x , r );
			ThreadPool::Parallel_for( 0 , dim , [&]( unsigned int thread , size_t i ){ r[i] = Pb[i] - r[i] , scratch[thread] += Dot( r[i] , r[i] ) , x[i] += (T)( d[i] * alpha ); } );
			for( unsigned int t=0 ; t<ThreadPool::NumThreads() ; t++ ) delta_new += scratch[t];
		}
		else
		{
			ThreadPool::Parallel_for( 0 , dim , [&]( unsigned int thread , size_t i ){ r[i] -=(T)( q[i] * alpha ) , scratch[thread] += Dot( r[i] , r[i] ) ,  x[i] += (T)( d[i] * alpha ); } );
			for( unsigned int t=0 ; t<ThreadPool::NumThreads() ; t++ ) delta_new += scratch[t];
		}

		Real beta = delta_new / delta_old;
		ThreadPool::Parallel_for( 0 , dim , [&]( unsigned int , size_t i ){ d[i] = r[i] + (T)( d[i] * beta ); } );
	}
	FreePointer( Pb );
	FreePointer( r );
	FreePointer( d );
	FreePointer( q );
	FreePointer( temp );
	return ii;
}
