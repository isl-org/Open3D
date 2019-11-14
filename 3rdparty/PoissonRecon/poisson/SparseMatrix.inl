/* -*- C++ -*-
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
#include <complex>
#include <unordered_map>

///////////////////////////////////////////////////////////////
//  SparseMatrix (unconstrained max row size specialization) //
///////////////////////////////////////////////////////////////
template< class T , class IndexType >
SparseMatrix< T , IndexType , 0 >::SparseMatrix( void )
{
	rowSizes = NullPointer( size_t );
	rowNum = 0;
	_entries = NullPointer( Pointer( MatrixEntry< T , IndexType > ) );
}

template< class T , class IndexType >
SparseMatrix< T , IndexType , 0 >::SparseMatrix( size_t rowNum )
{
	this->rowNum = 0;
	rowSizes = NullPointer( size_t );
	_entries= NullPointer( Pointer( MatrixEntry< T , IndexType > ) );
	resize( rowNum );
}
template< class T , class IndexType >
SparseMatrix< T , IndexType , 0 >::SparseMatrix( const SparseMatrix& M )
{
	rowSizes = NullPointer( size_t );
	rowNum = 0;
	_entries = NullPointer( Pointer( MatrixEntry< T , IndexType > ) );
	resize( M.rowNum );
	for( size_t i=0 ; i<rowNum ; i++ )
	{
		setRowSize( i , M.rowSizes[i] );
		for( size_t j=0 ; j<rowSizes[i] ; j++ ) _entries[i][j] = M._entries[i][j];
	}
}
template< class T , class IndexType >
SparseMatrix< T , IndexType , 0 >::SparseMatrix( SparseMatrix&& M )
{
	rowSizes = NullPointer( size_t );
	rowNum = 0;
	_entries = NullPointer( Pointer( MatrixEntry< T , IndexType > ) );

	Swap( *this , M );
}
template< class T , class IndexType >
template< class T2 , class IndexType2 >
SparseMatrix< T , IndexType , 0 >::SparseMatrix( const SparseMatrix< T2 , IndexType2 , 0 >& M )
{
	rowSizes = NullPointer( size_t );
	rowNum = 0;
	_entries = NULL;
	resize( M.rowNum );
	for( size_t i=0 ; i<rowNum ; i++ )
	{
		setRowSize( i , M.rowSizes[i] );
		for( size_t j=0 ; j<rowSizes[i] ; j++ ) _entries[i][j] = MatrixEntry< T , IndexType >( M._entries[i][j].N , T( M._entries[i][j].Value ) );
	}
}

template< class T , class IndexType >
template< class T2 , class IndexType2 >
SparseMatrix< T , IndexType , 0 >& SparseMatrix< T , IndexType , 0 >::copy( const SparseMatrix< T2 , IndexType2 , 0 >& M  )
{
	resize( M.rowNum );
	for ( size_t i=0 ; i<rowNum ; i++)
	{
		setRowSize( i , M.rowSizes[i] );
		for( size_t j=0 ; j<rowSizes[i] ; j++ )
		{
			IndexType2 idx = M._entries[i][j].N;
			_entries[i][j] = MatrixEntry< T , IndexType >( (IndexType)idx , T( M[i][j].Value ) );
		}
	}
	return *this;
}
template< class T , class IndexType >
SparseMatrix< T , IndexType , 0 >& SparseMatrix< T , IndexType , 0 >::operator = ( SparseMatrix< T , IndexType , 0 >&& M )
{
	Swap( *this , M );
	return *this;
}

template< class T , class IndexType >
SparseMatrix< T , IndexType , 0 >& SparseMatrix< T , IndexType , 0 >::operator = ( const SparseMatrix< T , IndexType , 0 >& M )
{
	resize( M.rowNum );
	for( size_t i=0 ; i<rowNum ; i++ )
	{
		setRowSize( i , M.rowSizes[i] );
		for( size_t j=0 ; j<rowSizes[i] ; j++ ) _entries[i][j]=M._entries[i][j];
	}
	return *this;
}
template< class T , class IndexType >
template< class T2 , class IndexType2 >
SparseMatrix< T , IndexType , 0 >& SparseMatrix< T , IndexType , 0 >::operator = (const SparseMatrix< T2 , IndexType2 , 0 >& M)
{
	resize( M.rowNum );
	for( size_t i=0 ; i<rowNum ; i++ )
	{
		setRowSize( i , M.rowSizes[i] );
		for( size_t j=0 ; j<rowSizes[i] ; j++ ) _entries[i][j] = MatrixEntry< T , IndexType >( M._entries[i][j].N , T( M._entries[i][j].Value ) );
	}
	return *this;
}

template< class T , class IndexType >
template< class T2 >
void SparseMatrix< T , IndexType , 0 >::operator() ( const T2* in , T2* out ) const { Interface::multiply( in , out ); }

template< class T , class IndexType > SparseMatrix< T , IndexType , 0 >::~SparseMatrix( void ) { resize( 0 ); }

template< class T , class IndexType >
void SparseMatrix< T , IndexType , 0 >::resize( size_t r )
{
	if( rowNum>0 )
	{
		for( size_t i=0 ; i<rowNum ; i++ ) FreePointer( _entries[i] );
		FreePointer( _entries );
		FreePointer( rowSizes );
	}
	rowNum = r;
	if( r )
	{
		rowSizes = AllocPointer< size_t >( r ) , memset( rowSizes , 0 , sizeof(size_t)*r );
		_entries = AllocPointer< Pointer( MatrixEntry< T , IndexType > ) >( r );
		for( size_t i=0 ; i<r ; i++ ) _entries[i] = NullPointer( MatrixEntry< T , IndexType > );
	}
}

template< class T , class IndexType >
void SparseMatrix< T , IndexType , 0 >::setRowSize( size_t row , size_t count )
{
	if( row>=0 && row<rowNum )
	{
		FreePointer( _entries[row] );
		if( count>0 )
		{
			_entries[ row ] = AllocPointer< MatrixEntry< T , IndexType > >( count );
			memset( _entries[ row ] , 0 , sizeof( MatrixEntry< T , IndexType > )*count );
		}
		rowSizes[row] = count;
	}
	else ERROR_OUT( "Row is out of bounds: 0 <= " , row , " < " , rowNum );
}
template< class T , class IndexType >
void SparseMatrix< T , IndexType , 0 >::resetRowSize( size_t row , size_t count )
{
	if( row>=0 && row<rowNum )
	{
		size_t oldCount = rowSizes[row];
		_entries[row] = ReAllocPointer< MatrixEntry< T, IndexType > >( _entries[row] , count );
		if( count>oldCount ) memset( _entries[row]+oldCount , 0 , sizeof( MatrixEntry< T , IndexType > ) * ( count - oldCount ) );
		rowSizes[row] = count;
	}
	else ERROR_OUT( "Row is out of bounds: 0 <= " , row , " < " , rowNum );
}

template< class T , class IndexType >
SparseMatrix< T , IndexType , 0 > SparseMatrix< T , IndexType , 0 >::Identity( size_t dim )
{
	SparseMatrix I;
	I.resize( dim );
	for( size_t i=0 ; i<dim ; i++ ) I.setRowSize( i , 1 ) , I[i][0] = MatrixEntry< T , IndexType >( (IndexType)i , (T)1 );
	return I;
}
template< class T , class IndexType >
SparseMatrix< T , IndexType , 0 >& SparseMatrix< T , IndexType , 0 >::operator *= ( T s )
{
	ThreadPool::Parallel_for( 0 , rowNum , [&]( unsigned int , size_t i ){ for( size_t j=0 ; j<rowSizes[i] ; j++ ) _entries[i][j].Value *= s; } );
	return *this;
}
template< class T , class IndexType >
SparseMatrix< T , IndexType , 0 >& SparseMatrix< T , IndexType , 0 >::operator /= ( T s ){ return (*this) * ( (T)1./s ); }
template< class T , class IndexType >
SparseMatrix< T , IndexType , 0 >& SparseMatrix< T , IndexType , 0 >::operator *= ( const SparseMatrix< T , IndexType , 0 >& B )
{
	SparseMatrix foo = (*this) * B;
	(*this) = foo;
	return *this;
}
template< class T , class IndexType >
SparseMatrix< T , IndexType , 0 >& SparseMatrix< T , IndexType , 0 >::operator += ( const SparseMatrix< T , IndexType , 0 >& B )
{
	SparseMatrix foo = (*this) + B;
	(*this) = foo;
	return *this;
}
template< class T , class IndexType >
SparseMatrix< T , IndexType , 0 >& SparseMatrix< T , IndexType , 0 >::operator -= ( const SparseMatrix< T , IndexType , 0 >& B )
{
	SparseMatrix foo = (*this) - B;
	(*this) = foo;
	return *this;
}
template< class T , class IndexType >
SparseMatrix< T , IndexType , 0 > SparseMatrix< T , IndexType , 0 >::operator * ( T s ) const
{
	SparseMatrix out = (*this);
	return out *= s;
}
template< class T , class IndexType >
SparseMatrix< T , IndexType , 0 > SparseMatrix< T , IndexType , 0 >::operator / ( T s ) const { return (*this) * ( (T)1. / s ); }
template< class T , class IndexType >
SparseMatrix< T , IndexType , 0 > SparseMatrix< T , IndexType , 0 >::operator * ( const SparseMatrix< T , IndexType , 0 >& B ) const
{
	SparseMatrix out;
	const SparseMatrix& A = *this;
	size_t aCols = 0 , aRows = A.rowNum;
	size_t bCols = 0 , bRows = B.rowNum;
	for( size_t i=0 ; i<A.rowNum ; i++ ) for( size_t j=0 ; j<A.rowSizes[i] ; j++ ) if( aCols<=A[i][j].N ) aCols = A[i][j].N+1;
	for( size_t i=0 ; i<B.rowNum ; i++ ) for( size_t j=0 ; j<B.rowSizes[i] ; j++ ) if( bCols<=B[i][j].N ) bCols = B[i][j].N+1;
	if( bRows<aCols ) ERROR_OUT( "Matrix sizes do not support multiplication " , aRows , " x " , aCols , " * " , bRows , " x " , bCols );

	out.resize( aRows );
	ThreadPool::Parallel_for( 0 , aRows , [&]( unsigned int , size_t i )
	{
		std::unordered_map< IndexType , T > row;
		for( size_t j=0 ; j<A.rowSizes[i] ; j++ )
		{
			IndexType idx1 = A[i][j].N;
			T AValue = A[i][j].Value;
			for( size_t k=0 ; k<B.rowSizes[idx1] ; k++ )
			{
				IndexType idx2 = B[idx1][k].N;
				T BValue = B[idx1][k].Value;
				typename std::unordered_map< IndexType , T >::iterator iter = row.find(idx2);
				if( iter==row.end() ) row[idx2] = AValue * BValue;
				else iter->second += AValue * BValue;
			}
		}
		out.setRowSize( i , row.size() );
		out.rowSizes[i] = 0;
		for( typename std::unordered_map< IndexType , T >::iterator iter=row.begin() ; iter!=row.end() ; iter++ ) out[i][ out.rowSizes[i]++ ] = MatrixEntry< T , IndexType >( iter->first , iter->second );
	}
	);
	return out;
}
template< class T , class IndexType >
SparseMatrix< T , IndexType , 0 > SparseMatrix< T , IndexType , 0 >::operator + ( const SparseMatrix< T , IndexType , 0 >& B ) const
{
	const SparseMatrix& A = *this;
	size_t rowNum = std::max< size_t >( A.rowNum , B.rowNum );
	SparseMatrix out;

	out.resize( rowNum );
	ThreadPool::Parallel_for( 0 , rowNum , [&]( unsigned int , size_t i )
	{
		std::unordered_map< IndexType , T > row;
		if( i<A.rowNum )
			for( size_t j=0 ; j<A.rowSizes[i] ; j++ )
			{
				IndexType idx = A[i][j].N;
				typename std::unordered_map< IndexType , T >::iterator iter = row.find(idx);
				if( iter==row.end() ) row[idx] = A[i][j].Value;
				else iter->second += A[i][j].Value;
			}
		if( i<B.rowNum )
			for( size_t j=0 ; j<B.rowSizes[i] ; j++ )
			{
				IndexType idx = B[i][j].N;
				typename std::unordered_map< IndexType , T >::iterator iter = row.find(idx);
				if( iter==row.end() ) row[idx] = B[i][j].Value;
				else iter->second += B[i][j].Value;
			}
		out.setRowSize( i , row.size() );
		out.rowSizes[i] = 0;
		for( typename std::unordered_map< IndexType , T >::iterator iter=row.begin() ; iter!=row.end() ; iter++ ) out[i][ out.rowSizes[i]++ ] = MatrixEntry< T , IndexType >( iter->first , iter->second );
	}
	);
	return out;
}
template< class T , class IndexType >
SparseMatrix< T , IndexType , 0 > SparseMatrix< T , IndexType , 0 >::operator - ( const SparseMatrix< T , IndexType , 0 >& B ) const
{
	const SparseMatrix& A = *this;
	size_t rowNum = std::max< size_t >( A.rowNum , B.rowNum );
	SparseMatrix out;

	out.resize( rowNum );
	ThreadPool::Parallel_for( 0 , rowNum , [&]( unsigned int , size_t i )
	{
		std::unordered_map< IndexType , T > row;
		if( i<A.rowNum )
			for( size_t j=0 ; j<A.rowSizes[i] ; j++ )
			{
				IndexType idx = A[i][j].N;
				typename std::unordered_map< IndexType , T >::iterator iter = row.find(idx);
				if( iter==row.end() ) row[idx] = A[i][j].Value;
				else iter->second += A[i][j].Value;
			}
		if( i<B.rowNum )
			for( size_t j=0 ; j<B.rowSizes[i] ; j++ )
			{
				IndexType idx = B[i][j].N;
				typename std::unordered_map< IndexType , T >::iterator iter = row.find(idx);
				if( iter==row.end() ) row[idx] = -B[i][j].Value;
				else iter->second -= B[i][j].Value;
			}
		out.setRowSize( i , row.size() );
		out.rowSizes[i] = 0;
		for( typename std::unordered_map< IndexType , T >::iterator iter=row.begin() ; iter!=row.end() ; iter++ ) out[i][ out.rowSizes[i]++ ] = MatrixEntry< T , IndexType >( iter->first , iter->second );
	}
	);
	return out;
}

template< class T , class IndexType >
SparseMatrix< T , IndexType , 0 > SparseMatrix< T , IndexType , 0 >::transpose( T (*TransposeFunction)( const T& ) ) const
{
	SparseMatrix A;
	const SparseMatrix& At = *this;
	size_t aRows = 0 , aCols = At.rowNum;
	for( size_t i=0 ; i<At.rowNum ; i++ ) for( size_t j=0 ; j<At.rowSizes[i] ; j++ ) if( aRows<=At[i][j].N ) aRows = At[i][j].N+1;

	A.resize( aRows );
	const size_t One = 1;
	for( size_t i=0 ; i<aRows ; i++ ) A.rowSizes[i] = 0;
	ThreadPool::Parallel_for( 0 , At.rowNum , [&]( unsigned int , size_t i )
	{
		for( size_t j=0 ; j<At.rowSizes[i] ; j++ )
		{
			AddAtomic( A.rowSizes[ At[i][j].N ] , One );
		}
	}
	);

	ThreadPool::Parallel_for( 0 , A.rowNum , [&]( unsigned int , size_t i )
	{
		size_t t = A.rowSizes[i];
		A.rowSizes[i] = 0;
		A.setRowSize( i , t );
		A.rowSizes[i] = 0;
	}
	);
	if( TransposeFunction ) for( size_t i=0 ; i<At.rowNum ; i++ ) for( size_t j=0 ; j<At.rowSizes[i] ; j++ )
	{
		size_t ii = At[i][j].N;
		A[ii][ A.rowSizes[ii]++ ] = MatrixEntry< T , IndexType >( (IndexType)i , TransposeFunction( At[i][j].Value ) );
	}
	else for( size_t i=0 ; i<At.rowNum ; i++ ) for( size_t j=0 ; j<At.rowSizes[i] ; j++ )
	{
		size_t ii = At[i][j].N;
		A[ii][ A.rowSizes[ii]++ ] = MatrixEntry< T , IndexType >( (IndexType)i , At[i][j].Value );
	}
	return A;
}
template< class T , class IndexType >
SparseMatrix< T , IndexType , 0 > SparseMatrix< T , IndexType , 0 >::transpose( size_t aRows , T (*TransposeFunction)( const T& ) ) const
{
	SparseMatrix A;
	const SparseMatrix& At = *this;
	size_t _aRows = 0 , aCols = At.rowNum;
	for( size_t i=0 ; i<At.rowNum ; i++ ) for( size_t j=0 ; j<At.rowSizes[i] ; j++ ) if( aCols<=At[i][j].N ) _aRows = At[i][j].N+1;
	if( _aRows>aRows ) ERROR_OUT( "Prescribed output dimension too low: " , aRows , " < " , _aRows );

	A.resize( aRows );
	for( size_t i=0 ; i<aRows ; i++ ) A.rowSizes[i] = 0;
	ThreadPool::Parallel_for( 0 , At.rowNum , [&]( unsigned int , size_t i )
	{
		for( size_t j=0 ; j<At.rowSizes[i] ; j++ ) AddAtomic( A.rowSizes[ At[i][j].N ] , 1 );
	}
	);

	ThreadPool::Parallel_for( 0 , A.rowNum , [&]( unsigned int , size_t i )
	{
		size_t t = A.rowSizes[i];
		A.rowSizes[i] = 0;
		A.setRowSize( i , t );
		A.rowSizes[i] = 0;
	}
	);
	if( TransposeFunction )
		for( size_t i=0 ; i<At.rowNum ; i++ ) for( size_t j=0 ; j<At.rowSizes[i] ; j++ )
		{
			size_t ii = At[i][j].N;
			A[ii][ A.rowSizes[ii]++ ] = MatrixEntry< T , IndexType >( (IndexType)i , TransposeFunction( At[i][j].Value ) );
		}
	else
		for( size_t i=0 ; i<At.rowNum ; i++ ) for( size_t j=0 ; j<At.rowSizes[i] ; j++ )
		{
			size_t ii = At[i][j].N;
			A[ii][ A.rowSizes[ii]++ ] = MatrixEntry< T , IndexType >( (IndexType)i , At[i][j].Value );
		}
	return A;
}

template< class T , class IndexType >
template< class A_const_iterator , class B_const_iterator >
SparseMatrix< T , IndexType , 0 > SparseMatrix< T , IndexType , 0 >::Multiply( const SparseMatrixInterface< T , A_const_iterator >& A , const SparseMatrixInterface< T , B_const_iterator >& B )
{
	SparseMatrix M;
	size_t aCols = 0 , aRows = A.rows();
	size_t bCols = 0 , bRows = B.rows();
	for( size_t i=0 ; i<A.rows() ; i++ ) for( A_const_iterator iter=A.begin(i) ; iter!=A.end(i) ; iter++ ) if( aCols<=iter->N ) aCols = iter->N+1;
	for( size_t i=0 ; i<B.rows() ; i++ ) for( B_const_iterator iter=B.begin(i) ; iter!=B.end(i) ; iter++ ) if( bCols<=iter->N ) bCols = iter->N+1;
	if( bRows<aCols ) ERROR_OUT( "Matrix sizes do not support multiplication " , aRows , " x " , aCols , " * " , bRows , " x " , bCols );

	M.resize( aRows );

	ThreadPool::Parallel_for( 0 , aRows , [&]( unsigned int , size_t i )
	{
		std::unordered_map< IndexType , T > row;
		for( A_const_iterator iterA=A.begin(i) ; iterA!=A.end(i) ; iterA++ )
		{
			IndexType idx1 = iterA->N;
			T AValue = iterA->Value;
			for( B_const_iterator iterB=B.begin(idx1) ; iterB!=B.end(idx1) ; iterB++ )
			{
				IndexType idx2 = iterB->N;
				T BValue = iterB->Value;
				T temp = BValue * AValue; // temp = A( i , idx1 ) * B( idx1 , idx2 )
				typename std::unordered_map< IndexType , T >::iterator iter = row.find(idx2);
				if( iter==row.end() ) row[idx2] = temp;
				else iter->second += temp;
			}
		}
		M.setRowSize( i , row.size() );
		M.rowSizes[i] = 0;
		for( typename std::unordered_map< IndexType , T >::iterator iter=row.begin() ; iter!=row.end() ; iter++ )
			M[i][ M.rowSizes[i]++ ] = MatrixEntry< T , IndexType >( iter->first , iter->second );
	}
	);
	return M;
}
template< class T , class IndexType >
template< class const_iterator >
SparseMatrix< T , IndexType , 0 > SparseMatrix< T , IndexType , 0 >::Transpose( const SparseMatrixInterface< T , const_iterator >& At , T (*TransposeFunction)( const T& ) )
{
	SparseMatrix< T , IndexType , 0 > A;
	size_t aRows = 0 , aCols = At.rows();
	for( size_t i=0 ; i<At.rows() ; i++ ) for( const_iterator iter=At.begin(i) ; iter!=At.end(i) ; iter++ ) if( aRows<=iter->N ) aRows = iter->N+1;

	A.resize( aRows );
	for( size_t i=0 ; i<aRows ; i++ ) A.rowSizes[i] = 0;
	for( size_t i=0 ; i<At.rows() ; i++ ) for( const_iterator iter=At.begin(i) ; iter!=At.end(i) ; iter++ ) A.rowSizes[ iter->N ]++;
	for( size_t i=0 ; i<A.rows ; i++ )
	{
		size_t t = A.rowSizes[i];
		A.rowSizes[i] = 0;
		A.setRowSize( i , t );
		A.rowSizes[i] = 0;
	}
	if( TransposeFunction )
		for( size_t i=0 ; i<At.rows() ; i++ ) for( const_iterator iter=At.begin(i) ; iter!=At.end(i) ; iter++ )
		{
			size_t ii = (size_t)iter->N;
			A[ii][ A.rowSizes[ii]++ ] = MatrixEntry< T , IndexType >( (IndexType)i , TransposeFunction( iter->Value ) );
		}
	else
		for( size_t i=0 ; i<At.rows() ; i++ ) for( const_iterator iter=At.begin(i) ; iter!=At.end(i) ; iter++ )
		{
			size_t ii = (size_t)iter->N;
			A[ii][ A.rowSizes[ii]++ ] = MatrixEntry< T , IndexType >( (IndexType)i , iter->Value );
		}
	return A;
}
template< class T , class IndexType >
template< class const_iterator >
SparseMatrix< T , IndexType , 0 > SparseMatrix< T , IndexType , 0 >::Transpose( const SparseMatrixInterface< T , const_iterator >& At , size_t outRows , T (*TransposeFunction)( const T& ) )
{
	SparseMatrix< T , IndexType , 0 > A;
	size_t _aRows = 0 , aCols = At.rows() , aRows = outRows;
	for( size_t i=0 ; i<At.rows() ; i++ ) for( const_iterator iter=At.begin(i) ; iter!=At.end(i) ; iter++ ) if( aCols<=iter->N ) _aRows = iter->N+1;
	if( _aRows>aRows ) ERROR_OUT( "Prescribed output dimension too low: " , aRows , " < " , _aRows );

	A.resize( aRows );
	for( size_t i=0 ; i<aRows ; i++ ) A.rowSizes[i] = 0;
	for( size_t i=0 ; i<At.rows() ; i++ ) for( const_iterator iter=At.begin(i) ; iter!=At.end(i) ; iter++ ) A.rowSizes[ iter->N ]++;
	for( size_t i=0 ; i<A.rows ; i++ )
	{
		size_t t = A.rowSizes[i];
		A.rowSizes[i] = 0;
		A.setRowSize( i , t );
		A.rowSizes[i] = 0;
	}
	if( TransposeFunction )
		for( size_t i=0 ; i<At.rows() ; i++ ) for( const_iterator iter=At.begin(i) ; iter!=At.end(i) ; iter++ )
		{
			size_t ii = (size_t)iter->N;
			A[ii][ A.rowSizes[ii]++ ] = MatrixEntry< T , IndexType >( (IndexType)i , TransposeFunction( iter->Value ) );
		}
	else
		for( size_t i=0 ; i<At.rows() ; i++ ) for( const_iterator iter=At.begin(i) ; iter!=At.end(i) ; iter++ )
		{
			size_t ii = (size_t)iter->N;
			A[ii][ A.rowSizes[ii]++ ] = MatrixEntry< T , IndexType >( (IndexType)i , iter->Value );
		}
	return true;
}

///////////////////////////////////////////
//  SparseMatrix (bounded max row size ) //
///////////////////////////////////////////

template< class T , class IndexType , size_t MaxRowSize >
SparseMatrix< T , IndexType , MaxRowSize >::SparseMatrix( void )
{
	_rowSizes = NullPointer( size_t );
	_rowNum = 0;
	_entries = NullPointer( MatrixEntry< T , IndexType > );
	_maxRows = 0;
}

template< class T , class IndexType , size_t MaxRowSize >
SparseMatrix< T , IndexType , MaxRowSize >::SparseMatrix( size_t rowNum ) : SparseMatrix()
{
	resize( rowNum );
}
template< class T , class IndexType , size_t MaxRowSize >
SparseMatrix< T , IndexType , MaxRowSize >::SparseMatrix( const SparseMatrix& M ) : SparseMatrix()
{
	resize( M._rowNum );
	for( size_t i=0 ; i<_rowNum ; i++ )
	{
		_rowSizes[i] = M._rowSizes[i];
		for( size_t j=0 ; j<_rowSizes[i] ; j++ ) _entries[ i + MaxRowSize*j ] = M._rowEntries[ i + MaxRowSize*j ];
	}
}
template< class T , class IndexType , size_t MaxRowSize >
SparseMatrix< T , IndexType , MaxRowSize >::SparseMatrix( SparseMatrix&& M ) : SparseMatrix()
{
	Swap( *this , M );
}
template< class T , class IndexType , size_t MaxRowSize >
template< class T2 , class IndexType2 >
SparseMatrix< T , IndexType , MaxRowSize >::SparseMatrix( const SparseMatrix< T2 , IndexType2 , MaxRowSize >& M ) : SparseMatrix()
{
	resize( M._rowNum );
	for( size_t i=0 ; i<_rowNum ; i++ )
	{
		_rowSizes[i] = M._rowSizes[i];
		for( size_t j=0 ; j<_rowSizes[i] ; j++ ) _entries[ i + MaxRowSize*j ] = MatrixEntry< T , IndexType >( M._rowEntries[i][j].N , T( M._entries[ i + MaxRowSize*j ].Value ) );
	}
}

template< class T , class IndexType , size_t MaxRowSize >
SparseMatrix< T , IndexType , MaxRowSize >& SparseMatrix< T , IndexType , MaxRowSize >::operator = ( SparseMatrix< T , IndexType , MaxRowSize >&& M )
{
	Swap( *this , M );
	return *this;
}

template< class T , class IndexType , size_t MaxRowSize >
SparseMatrix< T , IndexType , MaxRowSize >& SparseMatrix< T , IndexType , MaxRowSize >::operator = ( const SparseMatrix< T , IndexType , MaxRowSize >& M )
{
	resize( M._rowNum );
	for( size_t i=0 ; i<_rowNum ; i++ )
	{
		_rowSizes[i] = M._rowSizes[i];
		for( size_t j=0 ; j<_rowSizes[i] ; j++ ) _entries[ i + MaxRowSize*j ] = M._entries[ i + MaxRowSize*j ];
	}
	return *this;
}

template< class T , class IndexType , size_t MaxRowSize >
template< class T2 , class IndexType2 >
SparseMatrix< T , IndexType , MaxRowSize >& SparseMatrix< T , IndexType , MaxRowSize >::operator = ( const SparseMatrix< T2 , IndexType2 , MaxRowSize >& M )
{
	resize( M._rowNum );
	for( size_t i=0 ; i<_rowNum ; i++ )
	{
		_rowSizes[i] = M._rowSizes[i];
		for( size_t j=0 ; j<_rowSizes[i] ; j++ ) _entries[ i + MaxRowSize*j ] = MatrixEntry< T , IndexType >( M._entries[ i + MaxRowSize*j ].N , T( M._entries[ i + MaxRowSize*j ].Value ) );
	}
	return *this;
}

template< class T , class IndexType , size_t MaxRowSize >
template< class T2 >
void SparseMatrix< T , IndexType , MaxRowSize >::operator() ( const T2* in , T2* out ) const { Interface::multiply( in , out ); }

template< class T , class IndexType , size_t MaxRowSize >
SparseMatrix< T , IndexType , MaxRowSize >::~SparseMatrix( void )
{
	FreePointer( _rowSizes );
	FreePointer( _entries );
}

template< class T , class IndexType , size_t MaxRowSize >
void SparseMatrix< T , IndexType , MaxRowSize >::resize( size_t rowNum )
{
	_rowNum = rowNum;
	if( rowNum>_maxRows )
	{
		FreePointer( _rowSizes );
		FreePointer( _entries );

		if( rowNum )
		{
			_rowSizes = AllocPointer< size_t >( rowNum ) , memset( _rowSizes , 0 , sizeof(size_t)*rowNum );
			_entries = AllocPointer< MatrixEntry< T , IndexType > >( rowNum * MaxRowSize );
			_maxRows = rowNum;
		}
	}
}

template< class T , class IndexType , size_t MaxRowSize >
void SparseMatrix< T , IndexType , MaxRowSize >::setRowSize( size_t row , size_t rowSize )
{
	if( row>=_rowNum ) ERROR_OUT( "Row is out of bounds: 0 <= " , row , " < " , _rowNum );
	else if( rowSize>MaxRowSize ) ERROR_OUT( "Row size larger than max row size: " , rowSize , " < " , MaxRowSize );
	else _rowSizes[row] = rowSize;
}
template< class T , class IndexType , size_t MaxRowSize >
void SparseMatrix< T , IndexType , MaxRowSize >::resetRowSize( size_t row , size_t rowSize )
{
	if( row>=_rowNum ) ERROR_OUT( "Row is out of bounds: 0 <= " , row , " < " , _rowNum );
	else if( rowSize>MaxRowSize ) ERROR_OUT( "Row size larger than max row size: " , rowSize , " < " , MaxRowSize );
	else _rowSizes[row] = rowSize;
}

template< class T , class IndexType , size_t MaxRowSize >
SparseMatrix< T , IndexType , MaxRowSize >& SparseMatrix< T , IndexType , MaxRowSize >::operator *= ( T s )
{
	ThreadPool::Parallel_for( 0 , _rowNum*MaxRowSize , [&]( unsigned int , size_t i ){ _entries[i].Value *= s; } );
	return *this;
}

template< class T , class IndexType , size_t MaxRowSize >
SparseMatrix< T , IndexType , MaxRowSize >& SparseMatrix< T , IndexType , MaxRowSize >::operator /= ( T s ){ return (*this) * ( (T)1./s ); }

template< class T , class IndexType , size_t MaxRowSize >
SparseMatrix< T , IndexType , MaxRowSize > SparseMatrix< T , IndexType , MaxRowSize >::operator * ( T s ) const
{
	SparseMatrix out = (*this);
	return out *= s;
}

template< class T , class IndexType , size_t MaxRowSize >
SparseMatrix< T , IndexType , MaxRowSize > SparseMatrix< T , IndexType , MaxRowSize >::operator / ( T s ) const { return (*this) * ( (T)1. / s ); }
