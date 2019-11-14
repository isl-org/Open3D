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
#ifndef SPARSE_MATRIX_INCLUDED
#define SPARSE_MATRIX_INCLUDED

#include "SparseMatrixInterface.h"
#include "Array.h"

template< class T , class IndexType , size_t MaxRowSize=0 > class SparseMatrix;

template< class T , class IndexType > class SparseMatrix< T , IndexType , 0 > : public SparseMatrixInterface< T , ConstPointer( MatrixEntry< T , IndexType > ) >
{
	template< class T2 , class IndexType2 , size_t MaxRowSize2 > friend class SparseMatrix;
	Pointer( Pointer( MatrixEntry< T , IndexType > ) ) _entries;
public:
	static void Swap( SparseMatrix& M1 , SparseMatrix& M2 )
	{
		std::swap( M1.rowNum , M2.rowNum );
		std::swap( M1.rowSizes , M2.rowSizes );
		std::swap( M1._entries , M2._entries );
	}
	typedef SparseMatrixInterface< T , ConstPointer( MatrixEntry< T , IndexType > ) > Interface;
	typedef ConstPointer( MatrixEntry< T , IndexType > ) RowIterator;

	size_t rowNum;
	Pointer( size_t ) rowSizes;

	SparseMatrix( void );
	SparseMatrix( const SparseMatrix& M );
	SparseMatrix( SparseMatrix&& M );
	template< class T2 , class IndexType2 >
	SparseMatrix( const SparseMatrix< T2 , IndexType2 , 0 >& M );
	~SparseMatrix();
	SparseMatrix& operator = ( SparseMatrix&& M );
	SparseMatrix< T , IndexType >& operator = ( const SparseMatrix< T , IndexType >& M );
	template< class T2 , class IndexType2 >
	SparseMatrix< T , IndexType , 0 >& operator = ( const SparseMatrix< T2 , IndexType2 , 0 >& M );

	template< class T2 > void operator()( const T2* in , T2* out ) const;

	template< class T2 , class IndexType2 >
	SparseMatrix< T , IndexType , 0 >& copy( const SparseMatrix< T2 , IndexType2 , 0 >& M );

	inline ConstPointer( MatrixEntry< T , IndexType > ) begin( size_t row ) const { return _entries[row]; }
	inline ConstPointer( MatrixEntry< T , IndexType > ) end  ( size_t row ) const { return _entries[row] + (unsigned long long)rowSizes[row]; }
	inline size_t rows                              ( void )       const { return rowNum; }
	inline size_t rowSize                           ( size_t idx ) const { return rowSizes[idx]; }

	SparseMatrix( size_t rowNum );
	void resize	( size_t rowNum );
	void setRowSize( size_t row , size_t count );
	void resetRowSize( size_t row , size_t count );
	inline      Pointer( MatrixEntry< T , IndexType > ) operator[] ( size_t idx )       { return _entries[idx]; }
	inline ConstPointer( MatrixEntry< T , IndexType > ) operator[] ( size_t idx ) const { return _entries[idx]; }
	
	// With copy move, these should be well-behaved from a memory perspective
	static SparseMatrix Identity( size_t dim );
	SparseMatrix transpose(                  T (*TransposeFunction)( const T& )=NULL ) const;
	SparseMatrix transpose( size_t outRows , T (*TransposeFunction)( const T& )=NULL ) const;
	SparseMatrix  operator *  ( T s ) const;
	SparseMatrix  operator /  ( T s ) const;
	SparseMatrix  operator *  ( const SparseMatrix& M ) const;
	SparseMatrix  operator +  ( const SparseMatrix& M ) const;
	SparseMatrix  operator -  ( const SparseMatrix& M ) const;
	SparseMatrix& operator *= ( T s );
	SparseMatrix& operator /= ( T s );
	SparseMatrix& operator *= ( const SparseMatrix& M );
	SparseMatrix& operator += ( const SparseMatrix& M );
	SparseMatrix& operator -= ( const SparseMatrix& M );

	template< class A_const_iterator , class B_const_iterator >
	static SparseMatrix Multiply( const SparseMatrixInterface< T , A_const_iterator >& A , const SparseMatrixInterface< T , B_const_iterator >& B );
	template< class const_iterator >
	static SparseMatrix Transpose( const SparseMatrixInterface< T , const_iterator >& At , T (*TransposeFunction)( const T& )=NULL );
	template< class const_iterator >
	static SparseMatrix Transpose( const SparseMatrixInterface< T , const_iterator >& At , size_t outRows , T (*TransposeFunction)( const T& )=NULL );
};

template< class T , class IndexType , size_t MaxRowSize > class SparseMatrix : public SparseMatrixInterface< T , ConstPointer( MatrixEntry< T , IndexType > ) >
{
	template< class T2 , class IndexType2 > friend class _SparseMatrix;
	Pointer( MatrixEntry< T , IndexType > ) _entries;
	size_t _rowNum;
	Pointer( size_t ) _rowSizes;
	size_t _maxRows;
public:
	static void Swap( SparseMatrix& M1 , SparseMatrix& M2 )
	{
		std::swap( M1._rowNum , M2._rowNum );
		std::swap( M1._rowSizes , M2._rowSizes );
		std::swap( M1._entries , M2._entries );
	}
	typedef SparseMatrixInterface< T , ConstPointer( MatrixEntry< T , IndexType > ) > Interface;
	typedef ConstPointer( MatrixEntry< T , IndexType > ) RowIterator;

	SparseMatrix( void );
	SparseMatrix( const SparseMatrix& M );
	SparseMatrix( SparseMatrix&& M );
	template< class T2 , class IndexType2 >
	SparseMatrix( const SparseMatrix< T2 , IndexType2 , MaxRowSize >& M );
	SparseMatrix& operator = ( SparseMatrix&& M );
	SparseMatrix< T , IndexType , MaxRowSize >& operator = ( const SparseMatrix< T , IndexType , MaxRowSize >& M );
	template< class T2 , class IndexType2 >
	SparseMatrix< T , IndexType , MaxRowSize >& operator = ( const SparseMatrix< T2 , IndexType2 , MaxRowSize >& M );
	~SparseMatrix( void );

	template< class T2 > void operator()( const T2* in , T2* out ) const;

	inline ConstPointer( MatrixEntry< T , IndexType > ) begin( size_t row ) const { return _entries + MaxRowSize * row; }
	inline ConstPointer( MatrixEntry< T , IndexType > ) end  ( size_t row ) const { return _entries + MaxRowSize * row + (unsigned long long)_rowSizes[row]; }
	inline size_t rows                              ( void )       const { return _rowNum; }
	inline size_t rowSize                           ( size_t idx ) const { return _rowSizes[idx]; }

	SparseMatrix( size_t rowNum );
	void resize	( size_t rowNum );
	void setRowSize( size_t row , size_t rowSize );
	void resetRowSize( size_t row , size_t rowSize );
	inline      Pointer( MatrixEntry< T , IndexType > ) operator[] ( size_t idx )       { return _entries + MaxRowSize * idx; }
	inline ConstPointer( MatrixEntry< T , IndexType > ) operator[] ( size_t idx ) const { return _entries + MaxRowSize * idx; }

	// With copy move, these should be well-behaved from a memory perspective
	SparseMatrix  operator *  ( T s ) const;
	SparseMatrix  operator /  ( T s ) const;
	SparseMatrix& operator *= ( T s );
	SparseMatrix& operator /= ( T s );
};

#include "SparseMatrix.inl"
#endif // SPARSE_MATRIX_INCLUDED
