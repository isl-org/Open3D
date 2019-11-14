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

//////////////////////////////////////////
// IterationDimension < WindowDimension //
//////////////////////////////////////////
template< unsigned int WindowDimension , unsigned int IterationDimensions , unsigned int CurrentIteration >
struct _WindowLoop
{
protected:
	static const int CurrentDimension = CurrentIteration + WindowDimension - IterationDimensions;
	friend struct WindowLoop< WindowDimension , IterationDimensions >;
	friend struct _WindowLoop< WindowDimension , IterationDimensions , CurrentIteration+1 >;

	template< typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void Run( int begin , int end , UpdateFunction& updateState , ProcessFunction& function , Windows ... w )
	{
		for( int i=begin ; i<end ; i++ ){ updateState( WindowDimension - CurrentDimension , i ) ; _WindowLoop< WindowDimension , IterationDimensions , CurrentIteration-1 >::Run( begin , end , updateState , function , w[i] ... ); }
	}
	template< typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void Run( const int* begin , const int* end , UpdateFunction& updateState , ProcessFunction& function , Windows ... w )
	{
		for( int i=begin[0] ; i<end[0] ; i++ ){ updateState( WindowDimension - CurrentDimension , i ) ; _WindowLoop< WindowDimension , IterationDimensions , CurrentIteration-1 >::Run( begin+1 , end+1 , updateState , function , w[i] ... ); }
	}
	template< unsigned int ... Begin , unsigned int ... End , typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void Run( UIntPack< Begin ... > begin , UIntPack< End ... > end , UpdateFunction& updateState , ProcessFunction& function , Windows ... w )
	{
		for( int i=UIntPack< Begin ... >::First ; i<UIntPack< End ... >::First ; i++ ){ updateState( WindowDimension - CurrentDimension , i ) ; _WindowLoop< WindowDimension , IterationDimensions , CurrentIteration-1 >::Run( typename UIntPack< Begin ... >::Rest() , typename UIntPack< End ... >::Rest() , updateState , function , w[i] ... ); }
	}

	template< typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void RunParallel( int begin , int end , UpdateFunction& updateState , ProcessFunction& function , Windows ... w )
	{
		ThreadPool::Parallel_for( begin , end , [&]( unsigned int thread , size_t i ){ updateState( thread , WindowDimension - CurrentDimension , i ) ; _WindowLoop< WindowDimension , IterationDimensions , CurrentIteration-1 >::RunThread( begin , end , thread , updateState , function , w[i] ... ); } );
	}
	template< typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void RunParallel( const int* begin , const int* end , UpdateFunction& updateState , ProcessFunction& function , Windows ... w )
	{
		ThreadPool::Parallel_for( begin[0] , end[0] , [&]( unsigned int thread , size_t i ){ updateState( thread , WindowDimension - CurrentDimension , i ) ; _WindowLoop< WindowDimension , IterationDimensions , CurrentIteration-1 >::RunThread( begin+1 , end+1 , thread , updateState , function , w[i] ... ); } );
	}
	template< unsigned int ... Begin , unsigned int ... End , typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void RunParallel( UIntPack< Begin ... > begin , UIntPack< End ... > end , UpdateFunction& updateState , ProcessFunction& function , Windows ... w )
	{
		ThreadPool::Parallel_for( UIntPack< Begin ... >::First , UIntPack< End ... >::First , [&]( unsigned int thread , size_t i ){ updateState( thread , WindowDimension - CurrentDimension , i ) ; _WindowLoop< WindowDimension , IterationDimensions , CurrentIteration-1 >::RunThread( typename UIntPack< Begin ... >::Rest() , typename UIntPack< End ... >::Rest() , thread , updateState , function , w[i] ... ); } );
	}

	template< typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void RunThread( int begin , int end , unsigned int thread , UpdateFunction& updateState , ProcessFunction& function , Windows ... w )
	{
		for( int i=begin ; i<end ; i++ ){ updateState( thread , WindowDimension - CurrentDimension , i ) ; _WindowLoop< WindowDimension , IterationDimensions , CurrentIteration-1 >::RunThread( begin , end , thread , updateState , function , w[i] ... ); }
	}
	template< typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void RunThread( const int* begin , const int* end , unsigned int thread , UpdateFunction& updateState , ProcessFunction& function , Windows ... w )
	{
		for( int i=begin[0] ; i<end[0] ; i++ ){ updateState( thread , WindowDimension - CurrentDimension , i ) ; _WindowLoop< WindowDimension , IterationDimensions , CurrentIteration-1 >::RunThread( begin+1 , end+1 , thread , updateState , function , w[i] ... ); }
	}
	template< unsigned int ... Begin , unsigned int ... End , typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void RunThread( UIntPack< Begin ... > begin , UIntPack< End ... > end , unsigned int thread , UpdateFunction& updateState , ProcessFunction& function , Windows ... w )
	{
		for( int i=UIntPack< Begin ... >::First ; i<UIntPack< End ... >::First ; i++ ){ updateState( thread , WindowDimension - CurrentDimension , i ) ; _WindowLoop< WindowDimension , IterationDimensions , CurrentIteration-1 >::RunThread( typename UIntPack< Begin ... >::Rest() , typename UIntPack< End ... >::Rest() , thread , updateState , function , w[i] ... ); }
	}
};
//////////////////////////////////////////
// IterationDimension = WindowDimension //
//////////////////////////////////////////
template< unsigned int WindowDimension , unsigned int CurrentIteration >
struct _WindowLoop< WindowDimension , WindowDimension , CurrentIteration >
{
protected:
	static const int IterationDimensions = WindowDimension;
	static const int CurrentDimension = CurrentIteration + WindowDimension - IterationDimensions;
	friend struct WindowLoop< WindowDimension , IterationDimensions >;
	friend struct _WindowLoop< WindowDimension , IterationDimensions , CurrentIteration+1 >;

	template< typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void Run( int begin , int end , UpdateFunction& updateState , ProcessFunction& function , Windows ... w )
	{
		for( int i=begin ; i<end ; i++ ){ updateState( WindowDimension - CurrentDimension , i ) ; _WindowLoop< WindowDimension , IterationDimensions , CurrentIteration-1 >::Run( begin , end , updateState , function , w[i] ... ); }
	}
	template< typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void Run( const int* begin , const int* end , UpdateFunction& updateState , ProcessFunction& function , Windows ... w )
	{
		for( int i=begin[0] ; i<end[0] ; i++ ){ updateState( WindowDimension - CurrentDimension , i ) ; _WindowLoop< WindowDimension , IterationDimensions , CurrentIteration-1 >::Run( begin+1 , end+1 , updateState , function , w[i] ... ); }
	}
	template< unsigned int ... Begin , unsigned int ... End , typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void Run( UIntPack< Begin ... > begin , UIntPack< End ... > end , UpdateFunction& updateState , ProcessFunction& function , Windows ... w )
	{
		for( int i=UIntPack< Begin ... >::First ; i<UIntPack< End ... >::First ; i++ ){ updateState( WindowDimension - CurrentDimension , i ) ; _WindowLoop< WindowDimension , IterationDimensions , CurrentIteration-1 >::Run( typename UIntPack< Begin ... >::Rest() , typename UIntPack< End ... >::Rest() , updateState , function , w[i] ... ); }
	}

	template< typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void RunParallel( int begin , int end , UpdateFunction& updateState , ProcessFunction& function , Windows ... w )
	{
		ThreadPool::Parallel_for( begin , end , [&]( unsigned int thread , size_t i ){ updateState( thread , WindowDimension - CurrentDimension , i ) ; _WindowLoop< WindowDimension , IterationDimensions , CurrentIteration-1 >::RunThread( begin , end , thread , updateState , function , w[i] ... ); } );
	}
	template< typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void RunParallel( const int* begin , const int* end , UpdateFunction& updateState , ProcessFunction& function , Windows ... w )
	{
		ThreadPool::Parallel_for( begin[0] , end[0] , [&]( unsigned int thread , size_t i ){ updateState( thread , WindowDimension - CurrentDimension , i ) ; _WindowLoop< WindowDimension , IterationDimensions , CurrentIteration-1 >::RunThread( begin+1 , end+1 , thread , updateState , function , w[i] ... ); } );
	}
	template< unsigned int ... Begin , unsigned int ... End , typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void RunParallel( UIntPack< Begin ... > begin , UIntPack< End ... > end , UpdateFunction& updateState , ProcessFunction& function , Windows ... w )
	{
		ThreadPool::Parallel_for( UIntPack< Begin ... >::First , UIntPack< End ... >::First , [&]( unsigned int thread , size_t i ){ updateState( thread , WindowDimension - CurrentDimension , i ) ; _WindowLoop< WindowDimension , IterationDimensions , CurrentIteration-1 >::RunThread( typename UIntPack< Begin ... >::Rest() , typename UIntPack< End ... >::Rest() , thread , updateState , function , w[i] ... ); } );
	}

	template< typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void RunThread( int begin , int end , unsigned int thread , UpdateFunction& updateState , ProcessFunction& function , Windows ... w )
	{
		for( int i=begin ; i<end ; i++ ){ updateState( thread , WindowDimension - CurrentDimension , i ) ; _WindowLoop< WindowDimension , IterationDimensions , CurrentIteration-1 >::RunThread( begin , end , thread , updateState , function , w[i] ... ); }
	}
	template< typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void RunThread( const int* begin , const int* end , unsigned int thread , UpdateFunction& updateState , ProcessFunction& function , Windows ... w )
	{
		for( int i=begin[0] ; i<end[0] ; i++ ){ updateState( thread , WindowDimension - CurrentDimension , i ) ; _WindowLoop< WindowDimension , IterationDimensions , CurrentIteration-1 >::RunThread( begin+1 , end+1 , thread , updateState , function , w[i] ... ); }
	}
	template< unsigned int ... Begin , unsigned int ... End , typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void RunThread( UIntPack< Begin ... > begin , UIntPack< End ... > end , unsigned int thread , UpdateFunction& updateState , ProcessFunction& function , Windows ... w )
	{
		for( int i=UIntPack< Begin ... >::First ; i<UIntPack< End ... >::First ; i++ ){ updateState( thread , WindowDimension - CurrentDimension , i ) ; _WindowLoop< WindowDimension , IterationDimensions , CurrentIteration-1 >::RunThread( typename UIntPack< Begin ... >::Rest() , typename UIntPack< End ... >::Rest() , thread , updateState , function , w[i] ... ); }
	}
};
///////////////////////////////////////////////////////////////////
// IterationDimension < WindowDimension and CurrentIteration = 1 //
///////////////////////////////////////////////////////////////////
template< unsigned int WindowDimension , unsigned int IterationDimensions >
struct _WindowLoop< WindowDimension , IterationDimensions , 1 >
{
protected:
	static const unsigned int CurrentIteration = 1;
	static const int CurrentDimension = CurrentIteration + WindowDimension - IterationDimensions;
	friend struct WindowLoop< WindowDimension , IterationDimensions >;
	friend struct _WindowLoop< WindowDimension , IterationDimensions , CurrentIteration+1 >;

	template< typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void Run( int begin , int end , UpdateFunction& updateState , ProcessFunction& function , Windows ... w )
	{
		for( int i=begin ; i<end ; i++ ){ updateState( WindowDimension - CurrentDimension , i ) ; function( w[i] ... ); }
	}
	template< typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void Run( const int* begin , const int* end , UpdateFunction& updateState , ProcessFunction& function , Windows ... w )
	{
		for( int i=begin[0] ; i<end[0] ; i++ ){ updateState( WindowDimension - CurrentDimension , i ) ; function( w[i] ... ); }
	}
	template< unsigned int ... Begin , unsigned int ... End , typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void Run( UIntPack< Begin ... > begin , UIntPack< End ... > end , UpdateFunction& updateState , ProcessFunction& function , Windows ... w )
	{
		for( int i=UIntPack< Begin ... >::First ; i<UIntPack< End ... >::First ; i++ ){ updateState( WindowDimension - CurrentDimension , i ) ; function( w[i] ... ); }
	}

	template< typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void RunParallel( int begin , int end , UpdateFunction& updateState , ProcessFunction& function , Windows ... w )
	{
		ThreadPool::Parallel_for( begin , end , [&]( unsigned int thread , size_t i ){ updateState( thread , WindowDimension - CurrentDimension , i ) ; function( thread , w[i] ... ); } );
	}
	template< typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void RunParallel( const int* begin , const int* end , UpdateFunction& updateState , ProcessFunction& function , Windows ... w )
	{
		ThreadPool::Parallel_for( begin[0] , end[0] , [&]( unsigned int thread , size_t i ){ updateState( thread , WindowDimension - CurrentDimension , i ) ; function( thread , w[i] ... ); } );
	}
	template< unsigned int ... Begin , unsigned int ... End , typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void RunParallel( UIntPack< Begin ... > begin , UIntPack< End ... > end , UpdateFunction& updateState , ProcessFunction& function , Windows ... w )
	{
		ThreadPool::Parallel_for( UIntPack< Begin ... >::First , UIntPack< End ... >::First , [&]( unsigned int thread , size_t i ){ updateState( thread , WindowDimension - CurrentDimension , i ) ; function( thread , w[i] ... ); } );
	}

	template< typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void RunThread( int begin , int end , unsigned int thread , UpdateFunction& updateState , ProcessFunction& function , Windows ... w )
	{
		for( int i=begin ; i<end ; i++ ){ updateState( thread , WindowDimension - CurrentDimension , i ) ; function( thread , w[i] ... ); }
	}
	template< typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void RunThread( const int* begin , const int* end , unsigned int thread , UpdateFunction& updateState , ProcessFunction& function , Windows ... w )
	{
		for( int i=begin[0] ; i<end[0] ; i++ ){ updateState( thread , WindowDimension - CurrentDimension , i ) ; function( thread , w[i] ... ); }
	}
	template< unsigned int ... Begin , unsigned int ... End , typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void RunThread( UIntPack< Begin ... > begin , UIntPack< End ... > end , unsigned int thread , UpdateFunction& updateState , ProcessFunction& function , Windows ... w )
	{
		for( int i=UIntPack< Begin ... >::First ; i<UIntPack< End ... >::First ; i++ ){ updateState( thread , WindowDimension - CurrentDimension , i ) ; function( thread , w[i] ... ); }
	}
};
///////////////////////////////////////////////////////////////////
// IterationDimension = WindowDimension and CurrentIteration = 1 //
///////////////////////////////////////////////////////////////////
template< unsigned int WindowDimension >
struct _WindowLoop< WindowDimension , WindowDimension , 1 >
{
protected:
	static const unsigned int CurrentIteration = 1;
	static const int IterationDimensions = WindowDimension;
	static const int CurrentDimension = CurrentIteration + WindowDimension - IterationDimensions;
	friend struct WindowLoop< WindowDimension , IterationDimensions >;
	friend struct _WindowLoop< WindowDimension , IterationDimensions , CurrentIteration+1 >;

	template< typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void Run( int begin , int end , UpdateFunction& updateState , ProcessFunction& function , Windows ... w )
	{
		for( int i=begin ; i<end ; i++ ){ updateState( WindowDimension - CurrentDimension , i ) ; function( w[i] ... ); }
	}
	template< typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void Run( const int* begin , const int* end , UpdateFunction& updateState , ProcessFunction& function , Windows ... w )
	{
		for( int i=begin[0] ; i<end[0] ; i++ ){ updateState( WindowDimension - CurrentDimension , i ) ; function( w[i] ... ); }
	}
	template< unsigned int ... Begin , unsigned int ... End , typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void Run( UIntPack< Begin ... > begin , UIntPack< End ... > end , UpdateFunction& updateState , ProcessFunction& function , Windows ... w )
	{
		for( int i=UIntPack< Begin ... >::First ; i<UIntPack< End ... >::First ; i++ ){ updateState( WindowDimension - CurrentDimension , i ) ; function( w[i] ... ); }
	}

	template< typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void RunParallel( int begin , int end , UpdateFunction& updateState , ProcessFunction& function , Windows ... w )
	{
		ThreadPool::Parallel_for( begin , end , [&]( unsigned int thread , size_t i ){ updateState( thread , WindowDimension - CurrentDimension , i ) ; function( thread , w[i] ... ); } );
	}
	template< typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void RunParallel( const int* begin , const int* end , UpdateFunction& updateState , ProcessFunction& function , Windows ... w )
	{
		ThreadPool::Parallel_for( begin[0] , end[0] , [&]( unsigned int thread , size_t i ){ updateState( thread , WindowDimension - CurrentDimension , i ) ; function( thread , w[i] ... ); } );
	}
	template< unsigned int ... Begin , unsigned int ... End , typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void RunParallel( UIntPack< Begin ... > begin , UIntPack< End ... > end , UpdateFunction& updateState , ProcessFunction& function , Windows ... w )
	{
		ThreadPool::Parallel_for( UIntPack< Begin ... >::First , UIntPack< End ... >::First , [&]( unsigned int thread , size_t i ){ updateState( thread , WindowDimension - CurrentDimension , i ) ; function( thread , w[i] ... ); } );
	}

	template< typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void RunThread( int begin , int end , unsigned int thread , UpdateFunction& updateState , ProcessFunction& function , Windows ... w )
	{
		for( int i=begin ; i<end ; i++ ){ updateState( thread , WindowDimension - CurrentDimension , i ) ; function( thread , w[i] ... ); }
	}
	template< typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void RunThread( const int* begin , const int* end , unsigned int thread , UpdateFunction& updateState , ProcessFunction& function , Windows ... w )
	{
		for( int i=begin[0] ; i<end[0] ; i++ ){ updateState( thread , WindowDimension - CurrentDimension , i ) ; function( thread , w[i] ... ); }
	}
	template< unsigned int ... Begin , unsigned int ... End , typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void RunThread( UIntPack< Begin ... > begin , UIntPack< End ... > end , unsigned int thread , UpdateFunction& updateState , ProcessFunction& function , Windows ... w )
	{
		for( int i=UIntPack< Begin ... >::First ; i<UIntPack< End ... >::First ; i++ ){ updateState( thread , WindowDimension - CurrentDimension , i ) ; function( thread , w[i] ... ); }
	}
};
/////////////////////////////////////////////////////////////////
// IterationDimension = WindowDimension = CurrentIteration = 1 //
////////////////////////////////////////////////////////////////
template<  >
struct _WindowLoop< 1 , 1 , 1 >
{
protected:
	static const unsigned int CurrentIteration = 1;
	static const int WindowDimension = 1;
	static const int IterationDimensions = WindowDimension;
	static const int CurrentDimension = CurrentIteration + WindowDimension - IterationDimensions;
	friend struct WindowLoop< WindowDimension , IterationDimensions >;
	friend struct _WindowLoop< WindowDimension , IterationDimensions , CurrentIteration+1 >;

	template< typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void Run( int begin , int end , UpdateFunction& updateState , ProcessFunction& function , Windows ... w )
	{
		for( int i=begin ; i<end ; i++ ){ updateState( WindowDimension - CurrentDimension , i ) ; function( w[i] ... ); }
	}
	template< typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void Run( const int* begin , const int* end , UpdateFunction& updateState , ProcessFunction& function , Windows ... w )
	{
		for( int i=begin[0] ; i<end[0] ; i++ ){ updateState( WindowDimension - CurrentDimension , i ) ; function( w[i] ... ); }
	}
	template< unsigned int ... Begin , unsigned int ... End , typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void Run( UIntPack< Begin ... > begin , UIntPack< End ... > end , UpdateFunction& updateState , ProcessFunction& function , Windows ... w )
	{
		for( int i=UIntPack< Begin ... >::First ; i<UIntPack< End ... >::First ; i++ ){ updateState( WindowDimension - CurrentDimension , i ) ; function( w[i] ... ); }
	}

	template< typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void RunParallel( int begin , int end , UpdateFunction& updateState , ProcessFunction& function , Windows ... w )
	{
		ThreadPool::Parallel_for( begin , end , [&]( unsigned int thread , size_t i ){ updateState( thread , WindowDimension - CurrentDimension , i ) ; function( thread , w[i] ... ); } );
	}
	template< typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void RunParallel( const int* begin , const int* end , UpdateFunction& updateState , ProcessFunction& function , Windows ... w )
	{
		ThreadPool::Parallel_for( begin[0] , end[0] , [&]( unsigned int thread , size_t i ){ updateState( thread , WindowDimension - CurrentDimension , i ) ; function( thread , w[i] ... ); } );
	}
	template< unsigned int ... Begin , unsigned int ... End , typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void RunParallel( UIntPack< Begin ... > begin , UIntPack< End ... > end , UpdateFunction& updateState , ProcessFunction& function , Windows ... w )
	{
		ThreadPool::Parallel_for( UIntPack< Begin ... >::First , UIntPack< End ... >::First , [&]( unsigned int thread , size_t i ){ updateState( thread , WindowDimension - CurrentDimension , i ) ; function( thread , w[i] ... ); } );
	}

	template< typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void RunThread( int begin , int end , unsigned int thread , UpdateFunction& updateState , ProcessFunction& function , Windows ... w )
	{
		for( int i=begin ; i<end ; i++ ){ updateState( thread , WindowDimension - CurrentDimension , i ) ; function( thread , w[i] ... ); }
	}
	template< typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void RunThread( const int* begin , const int* end , unsigned int thread , UpdateFunction& updateState , ProcessFunction& function , Windows ... w )
	{
		for( int i=begin[0] ; i<end[0] ; i++ ){ updateState( thread , WindowDimension - CurrentDimension , i ) ; function( thread , w[i] ... ); }
	}
	template< unsigned int ... Begin , unsigned int ... End , typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void RunThread( UIntPack< Begin ... > begin , UIntPack< End ... > end , unsigned int thread , UpdateFunction& updateState , ProcessFunction& function , Windows ... w )
	{
		for( int i=UIntPack< Begin ... >::First ; i<UIntPack< End ... >::First ; i++ ){ updateState( thread , WindowDimension - CurrentDimension , i ) ; function( thread , w[i] ... ); }
	}
};
//////////////////////////
// CurrentIteration = 0 //
//////////////////////////
template< unsigned int WindowDimension , unsigned int IterationDimensions >
struct _WindowLoop< WindowDimension , IterationDimensions , 0 >
{
protected:
	static const unsigned int CurrentIteration = 0;
	static const int CurrentDimension = CurrentIteration + WindowDimension - IterationDimensions;
	friend struct WindowLoop< WindowDimension , IterationDimensions >;
	friend struct _WindowLoop< WindowDimension , IterationDimensions , CurrentIteration+1 >;

	template< typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void Run( int begin , int end , UpdateFunction& updateState , ProcessFunction& function , Windows ... w ){ ERROR_OUT( "Shouldn't be here" ); }
	template< typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void Run( const int* begin , const int* end , UpdateFunction& updateState , ProcessFunction& function , Windows ... w ){ ERROR_OUT( "Shouldn't be here" ); }
	template< unsigned int ... Begin , unsigned int ... End , typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void Run( UIntPack< Begin ... > begin , UIntPack< End ... > end , UpdateFunction& updateState , ProcessFunction& function , Windows ... w ){ ERROR_OUT( "Shouldn't be here" ); }

	template< typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void RunParallel( int begin , int end , UpdateFunction& updateState , ProcessFunction& function , Windows ... w ){ ERROR_OUT( "Shouldn't be here" ); }
	template< typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void RunParallel( const int* begin , const int* end , UpdateFunction& updateState , ProcessFunction& function , Windows ... w ){ ERROR_OUT( "Shouldn't be here" ); }
	template< unsigned int ... Begin , unsigned int ... End , typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void RunParallel( UIntPack< Begin ... > begin , UIntPack< End ... > end , UpdateFunction& updateState , ProcessFunction& function , Windows ... w ){ ERROR_OUT( "Shouldn't be here" ); }

	template< typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void RunThread( int begin , int end , unsigned int thread , UpdateFunction& updateState , ProcessFunction& function , Windows ... w ){ ERROR_OUT( "Shouldn't be here" ); }
	template< typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void RunThread( const int* begin , const int* end , unsigned int thread , UpdateFunction& updateState , ProcessFunction& function , Windows ... w ){ ERROR_OUT( "Shouldn't be here" ); }
};
