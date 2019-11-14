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

#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include "MyMiscellany.h"

/////////////////////
// RegularTreeNode //
/////////////////////
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::RegularTreeNode( void )
{
	parent = children = NULL;
	_depth = 0;
	memset( _offset , 0 , sizeof(_offset ) );
}
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< typename Initializer >
RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::RegularTreeNode( Initializer &initializer ) : RegularTreeNode() { initializer( *this ); }

template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
void RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::cleanChildren( bool deleteChildren )
{
	if( children )
	{
		for( int c=0 ; c<(1<<Dim) ; c++ ) children[c].cleanChildren( deleteChildren );
		if( deleteChildren ) delete[] children;
	}
	parent = children = NULL;
}
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::~RegularTreeNode(void)
{
#ifdef SHOW_WARNINGS
#pragma message( "[WARNING] Deallocation of children is your responsibility" )
#endif // SHOW_WARNINGS
	parent = children = NULL;
}
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< typename Initializer >
RegularTreeNode< Dim , NodeData , DepthAndOffsetType >* RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::NewBrood( Allocator< RegularTreeNode >* nodeAllocator , Initializer &initializer )
{
	RegularTreeNode< Dim , NodeData , DepthAndOffsetType >* brood;
	if( nodeAllocator ) brood = nodeAllocator->newElements( 1<<Dim );
	else                brood = new RegularTreeNode[ 1<<Dim ];
	for( int idx=0 ; idx<(1<<Dim) ; idx++ )
	{
		initializer( brood[idx] );
		brood[idx]._depth = 0;
		for( int d=0 ; d<Dim ; d++ ) brood[idx]._offset[d] = (idx>>d) & 1;
	}
	return brood;
}
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
void RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::ResetDepthAndOffset( RegularTreeNode* root , int d , int off[Dim] )
{
	std::function< void ( int& , int[Dim] ) > ParentDepthAndOffset = [] ( int& d , int off[Dim] ){ d-- ; for( int _d=0 ; _d<Dim ; _d++ ) off[_d]>>=1 ; };
	std::function< void ( int& , int[Dim] ) >  ChildDepthAndOffset = [] ( int& d , int off[Dim] ){ d++ ; for( int _d=0 ; _d<Dim ; _d++ ) off[_d]<<=1 ; };
	std::function< RegularTreeNode* ( RegularTreeNode* , int& , int[] ) > _nextBranch = [&]( RegularTreeNode* current , int& d , int off[Dim] )
	{
		if( current==root ) return (RegularTreeNode*)NULL;
		else
		{
			int c = (int)( current - current->parent->children );

			if( c==(1<<Dim)-1 )
			{
				ParentDepthAndOffset( d , off );
				return _nextBranch( current->parent , d , off );
			}
			else
			{
				ParentDepthAndOffset( d , off ) ; ChildDepthAndOffset( d , off );
				for( int _d=0 ; _d<Dim ; _d++ ) off[_d] |= ( ( (c+1)>>_d ) & 1 );
				return current+1;
			}
		}
	};
	auto _nextNode = [&]( RegularTreeNode* current , int& d , int off[Dim] )
	{
		if( !current ) return root;
		else if( current->children )
		{
			ChildDepthAndOffset( d , off );
			return current->children;
		}
		else return _nextBranch( current , d , off );
	};
	for( RegularTreeNode* node=_nextNode( NULL , d , off ) ; node ; node = _nextNode( node , d , off ) )
	{
		node->_depth = (DepthAndOffsetType)d;
		for( int _d=0 ; _d<Dim ; _d++ ) node->_offset[_d] = (DepthAndOffsetType)off[_d];
	}
}

template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< typename Initializer >
void RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::setFullDepth( int maxDepth , Allocator< RegularTreeNode >* nodeAllocator , Initializer &initializer )
{
	if( maxDepth>0 )
	{
		if( !children ) initChildren< false >( nodeAllocator , initializer );
		for( int i=0 ; i<(1<<Dim) ; i++ ) children[i].setFullDepth( maxDepth-1 , nodeAllocator , initializer );
	}
}

template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< typename Initializer >
bool RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::_initChildren( Allocator< RegularTreeNode >* nodeAllocator , Initializer &initializer )
{
	if( nodeAllocator ) children = nodeAllocator->newElements( 1<<Dim );
	else
	{
		if( children ) delete[] children;
		children = new RegularTreeNode[ 1<<Dim ];
	}
	if( !children ) ERROR_OUT( "Failed to initialize children" );
	for( int idx=0 ; idx<(1<<Dim) ; idx++ )
	{
		children[idx].parent = this;
		children[idx].children = NULL;
		initializer( children[idx] );
		children[idx]._depth = _depth+1;
		for( int d=0 ; d<Dim ; d++ ) children[idx]._offset[d] = (_offset[d]<<1) | ( (idx>>d) & 1 );
	}
	return true;
}
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< typename Initializer >
bool RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::_initChildren_s( Allocator< RegularTreeNode >* nodeAllocator , Initializer &initializer )
{
	RegularTreeNode * volatile & children = this->children;
	RegularTreeNode *_children;

	// Allocate the children
	if( nodeAllocator ) _children = nodeAllocator->newElements( 1<<Dim );
	else                _children = new RegularTreeNode[ 1<<Dim ];
	if( !_children ) ERROR_OUT( "Failed to initialize children" );
	for( int idx=0 ; idx<(1<<Dim) ; idx++ )
	{
		_children[idx].parent = this;
		_children[idx].children = NULL;
		_children[idx]._depth = _depth+1;
		for( int d=0 ; d<Dim ; d++ ) _children[idx]._offset[d] = (_offset[d]<<1) | ( (idx>>d) & 1 );
		// [WARNING] We are assuming that it's OK to initialize nodes that may not be used.
		for( int idx=0 ; idx<(1<<Dim) ; idx++ ) initializer( _children[idx] );
	}

	// If we are the first to set the child, initialize
	if( SetAtomic( &children , _children , (RegularTreeNode *)NULL ) ) return true;
	// Otherwise clean up
	else
	{
		if( nodeAllocator ) ;
		else delete[] _children;
		return false;
	}
}

template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< class MergeFunctor >
void RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::merge( RegularTreeNode* node , MergeFunctor& f )
{
	if( node )
	{
		nodeData = f( nodeData , node->nodeData );
		if( children && node->children ) for( int c=0 ; c<(1<<Dim) ; c++ ) children[c].merge( node->children[c] , f );
		else if( node->children )
		{
			children = node->children;
			for( int c=0 ; c<(1<<Dim) ; c++ ) children[c].parent = this;
			node->children = NULL;
		}
	}
}

template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
inline void RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::depthAndOffset( int& depth , int offset[Dim] ) const
{
	depth = _depth;
	for( int d=0 ; d<Dim ; d++ ) offset[d] = _offset[d];
}
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
inline void RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::centerIndex( int index[Dim] ) const
{
	for( int i=0 ; i<Dim ; i++ ) index[i] = BinaryNode::CenterIndex( _depth , _offset[i] );
}
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
inline int RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::depth( void ) const { return _depth; }
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< class Real >
void RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::centerAndWidth( Point< Real , Dim >& center , Real& width ) const
{
	width = Real( 1.0 / (1<<_depth) );
	for( int d=0 ; d<Dim ; d++ ) center[d] = Real( 0.5+_offset[d] ) * width;
}
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< class Real >
void RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::startAndWidth( Point< Real , Dim >& start , Real& width ) const
{
	width = Real( 1.0 / (1<<_depth) );
	for( int d=0 ; d<Dim ; d++ ) start[d] = Real( _offset[d] ) * width;
}
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< class Real >
bool RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::isInside( Point< Real , Dim > p ) const
{
	Point< Real , Dim > c ; Real w;
	centerAndWidth( c , w ) , w /= 2;
	for( int d=0 ; d<Dim ; d++ ) if( p[d]<=(c[d]-w) || p[d]>(c[d]+w) ) return false;
	return true;
}

template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
int RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::maxDepth(void) const
{
	if( !children ) return 0;
	else
	{
		int c , d;
		for( int i=0 ; i<(1<<Dim) ; i++ )
		{
			d = children[i].maxDepth();
			if( !i || d>c ) c=d;
		}
		return c+1;
	}
}
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
size_t RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::nodes( void ) const
{
	if( !children ) return 1;
	else
	{
		size_t c=0;
		for( int i=0 ; i<(1<<Dim) ; i++ ) c += children[i].nodes();
		return c+1;
	}
}
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
size_t RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::leaves( void ) const
{
	if( !children ) return 1;
	else
	{
		size_t c=0;
		for( int i=0 ; i<(1<<Dim) ; i++ ) c += children[i].leaves();
		return c;
	}
}
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
size_t RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::maxDepthLeaves( int maxDepth ) const
{
	if( depth()>maxDepth ) return 0;
	if( !children ) return 1;
	else
	{
		size_t c=0;
		for( int i=0 ; i<(1<<Dim) ; i++ ) c += children[i].maxDepthLeaves(maxDepth);
		return c;
	}
}
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
const RegularTreeNode< Dim , NodeData , DepthAndOffsetType >* RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::root( void ) const
{
	const RegularTreeNode* temp = this;
	while( temp->parent ) temp = temp->parent;
	return temp;
}
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
const RegularTreeNode< Dim , NodeData , DepthAndOffsetType >* RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::nextBranch( const RegularTreeNode* current ) const
{
	if( !current->parent || current==this ) return NULL;
	if( current-current->parent->children==(1<<Dim)-1 ) return nextBranch( current->parent );
	else return current+1;
}
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
RegularTreeNode< Dim , NodeData , DepthAndOffsetType >* RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::nextBranch(RegularTreeNode* current){
	if( !current->parent || current==this ) return NULL;
	if( current-current->parent->children==(1<<Dim)-1 ) return nextBranch(current->parent);
	else return current+1;
}
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
const RegularTreeNode< Dim , NodeData , DepthAndOffsetType >* RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::prevBranch( const RegularTreeNode* current ) const
{
	if( !current->parent || current==this ) return NULL;
	if( current-current->parent->children==0 ) return prevBranch( current->parent );
	else return current-1;
}
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
RegularTreeNode< Dim , NodeData , DepthAndOffsetType >* RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::prevBranch( RegularTreeNode* current )
{
	if( !current->parent || current==this ) return NULL;
	if( current-current->parent->children==0 ) return prevBranch( current->parent );
	else return current-1;
}
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
const RegularTreeNode< Dim , NodeData , DepthAndOffsetType >* RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::nextLeaf(const RegularTreeNode* current) const{
	if(!current)
	{
		const RegularTreeNode< Dim , NodeData , DepthAndOffsetType >* temp=this;
		while( temp->children ) temp = temp->children;
		return temp;
	}
	if( current->children ) return current->nextLeaf();
	const RegularTreeNode* temp=nextBranch( current );
	if( !temp ) return NULL;
	else return temp->nextLeaf();
}
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
RegularTreeNode< Dim , NodeData , DepthAndOffsetType >* RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::nextLeaf(RegularTreeNode* current){
	if( !current )
	{
		RegularTreeNode< Dim , NodeData , DepthAndOffsetType >* temp=this;
		while( temp->children ) temp = temp->children;
		return temp;
	}
	if( current->children ) return current->nextLeaf();
	RegularTreeNode* temp=nextBranch( current) ;
	if( !temp ) return NULL;
	else return temp->nextLeaf();
}
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< typename NodeTerminationLambda >
const RegularTreeNode< Dim , NodeData , DepthAndOffsetType >* RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::nextNode( NodeTerminationLambda &ntl , const RegularTreeNode *current ) const
{
	if( !current ) return this;
	else if( current->children && !ntl(current) ) return current->children;
	else return nextBranch( current );
}
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< typename NodeTerminationLambda >
RegularTreeNode< Dim , NodeData , DepthAndOffsetType >* RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::nextNode( NodeTerminationLambda &ntl , RegularTreeNode* current )
{
	if( !current ) return this;
	else if( current->children && !ntl(current) ) return current->children;
	else return nextBranch( current );
}

template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
const RegularTreeNode< Dim , NodeData , DepthAndOffsetType >* RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::nextNode( const RegularTreeNode* current ) const
{
	if( !current ) return this;
	else if( current->children ) return current->children;
	else return nextBranch( current );
}
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
RegularTreeNode< Dim , NodeData , DepthAndOffsetType >* RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::nextNode( RegularTreeNode* current )
{
	if( !current ) return this;
	else if( current->children ) return current->children;
	else return nextBranch( current );
}

template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
void RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::printRange(void) const
{
	Point< float , Dim > center;
	float width;
	centerAndWidth( center , width );
	for( int d=0 ; d<Dim ; d++ )
	{
		printf( "[%f,%f]" , center[d]-width/2 , center[d]+width/2 );
		if( d<Dim-1 ) printf( " x " );
		else printf("\n");
	}
}

template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< class Real >
int RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::ChildIndex( const Point< Real , Dim >& center , const Point< Real , Dim >& p )
{
	int cIndex=0;
	for( int d=0 ; d<Dim ; d++ ) if( p[d]>center[d] ) cIndex |= (1<<d);
	return cIndex;
}

template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
bool RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::write( const char* fileName ) const
{
	FILE* fp=fopen( fileName , "wb" );
	if( !fp ) return false;
	bool ret = write(fp);
	fclose(fp);
	return ret;
}
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
bool RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::write( FILE* fp ) const
{
	fwrite( this , sizeof( RegularTreeNode< Dim , NodeData , DepthAndOffsetType > ) , 1 , fp );
	if( children ) for( int i=0 ; i<(1<<Dim) ; i++ ) children[i].write(fp);
	return true;
}
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< typename Initializer >
bool RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::read( const char* fileName , Allocator< RegularTreeNode >* nodeAllocator , Initializer &initializer )
{
	FILE* fp = fopen( fileName , "rb" );
	if( !fp ) return false;
	bool ret = read( fp , nodeAllocator , initializer );
	fclose( fp );
	return ret;
}
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< typename Initializer >
bool RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::read( FILE* fp , Allocator< RegularTreeNode >* nodeAllocator , Initializer &initializer )
{
	if( fread( this , sizeof( RegularTreeNode< Dim , NodeData , DepthAndOffsetType > ) , 1 , fp )!=1 ) ERROR_OUT( "Failed to read node" );
	parent = NULL;
	if( children )
	{
		children = NULL;
		initChildren< false >( nodeAllocator , initializer );
		for( int i=0 ; i<(1<<Dim) ; i++ ) children[i].read( fp , nodeAllocator , initializer ) , children[i].parent = this;
	}
	return true;
}
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
int RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::width( int maxDepth ) const
{
	int d=depth();
	return 1<<(maxDepth-d); 
}

////////////////////////////////
// RegularTreeNode::Neighbors //
////////////////////////////////
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... Widths >
RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::Neighbors< UIntPack< Widths ... > >::Neighbors( void ){ static_assert( sizeof...(Widths)==Dim , "[ERROR] Window and tree dimensions don't match" ) ; clear(); }
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... Widths >
void RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::Neighbors< UIntPack< Widths ... > >::clear( void ){ for( int i=0 ; i<WindowSize< UIntPack< Widths ... > >::Size ; i++ ) neighbors.data[i] = NULL; }

/////////////////////////////////////
// RegularTreeNode::ConstNeighbors //
/////////////////////////////////////
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... Widths >
RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::ConstNeighbors< UIntPack< Widths ... > >::ConstNeighbors( void ){ static_assert( sizeof...(Widths)==Dim , "[ERROR] Window and tree dimensions don't match" ) ; clear(); }
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... Widths >
void RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::ConstNeighbors< UIntPack< Widths ... > >::clear( void ){ for( int i=0 ; i<WindowSize< UIntPack< Widths ... > >::Size ; i++ ) neighbors.data[i] = NULL; }

//////////////////////////////////
// RegularTreeNode::NeighborKey //
//////////////////////////////////
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... LeftRadii , unsigned int ... RightRadii >
RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::NeighborKey< UIntPack< LeftRadii ... > , UIntPack< RightRadii ... > >::NeighborKey( void ){ _depth=-1 , neighbors=NULL; }
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... LeftRadii , unsigned int ... RightRadii >
RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::NeighborKey< UIntPack< LeftRadii ... > , UIntPack< RightRadii ... > >::NeighborKey( const NeighborKey& key )
{
	_depth = 0 , neighbors = NULL;
	set( key._depth );
	for( int d=0 ; d<=_depth ; d++ ) memcpy( &neighbors[d] , &key.neighbors[d] , sizeof( Neighbors< UIntPack< ( LeftRadii + RightRadii + 1 ) ... > > ) );
}
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... LeftRadii , unsigned int ... RightRadii >
RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::NeighborKey< UIntPack< LeftRadii ... > , UIntPack< RightRadii ... > >::~NeighborKey( void )
{
	if( neighbors ) delete[] neighbors;
	neighbors=NULL;
}
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... LeftRadii , unsigned int ... RightRadii >
void RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::NeighborKey< UIntPack< LeftRadii ... > , UIntPack< RightRadii ... > >::set( int d )
{
	if( neighbors ) delete[] neighbors;
	neighbors = NULL;
	_depth = d;
	if( d<0 ) return;
	neighbors = new NeighborType[d+1];
}
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... LeftRadii , unsigned int ... RightRadii >
template< bool CreateNodes , bool ThreadSafe , typename NodeInitializer , unsigned int ... _PLeftRadii , unsigned int ... _PRightRadii , unsigned int ... _CLeftRadii , unsigned int ... _CRightRadii >
unsigned int RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::NeighborKey< UIntPack< LeftRadii ... > , UIntPack< RightRadii ... > >::_NeighborsLoop( UIntPack< _PLeftRadii ... > pLeftRadii , UIntPack< _PRightRadii ... > pRightRadii , UIntPack< _CLeftRadii ... > cLeftRadii , UIntPack< _CRightRadii ... > cRightRadii , ConstWindowSlice< RegularTreeNode* , UIntPack< ( _PLeftRadii + _PRightRadii + 1 ) ... > > pNeighbors , WindowSlice< RegularTreeNode* , UIntPack< ( _CLeftRadii + _CRightRadii + 1 ) ... > > cNeighbors , int cIdx , Allocator< RegularTreeNode >* nodeAllocator , NodeInitializer &initializer )
{
	static_assert( Dim==sizeof ... ( _PLeftRadii ) && Dim==sizeof ... ( _PRightRadii ) && Dim==sizeof ... ( _CLeftRadii ) && Dim==sizeof ... ( _CRightRadii ) , "[ERROR] Dimensions don't match" );
	int c[Dim];
	for( int d=0 ; d<Dim ; d++ ) c[d] = ( cIdx>>d ) & 1;
	return _Run< CreateNodes , ThreadSafe , NodeInitializer , UIntPack< _PLeftRadii ... > , UIntPack< _PRightRadii ... > , UIntPack< _CLeftRadii ... > , UIntPack< _CRightRadii ... > >::Run( pNeighbors , cNeighbors , c , 0 , nodeAllocator , initializer );
}
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... LeftRadii , unsigned int ... RightRadii >
template< bool CreateNodes , bool ThreadSafe , typename NodeInitializer , unsigned int ... _PLeftRadii , unsigned int ... _PRightRadii , unsigned int ... _CLeftRadii , unsigned int ... _CRightRadii >
unsigned int RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::NeighborKey< UIntPack< LeftRadii ... > , UIntPack< RightRadii ... > >::_NeighborsLoop( UIntPack< _PLeftRadii ... > pLeftRadii , UIntPack< _PRightRadii ... > pRightRadii , UIntPack< _CLeftRadii ... > cLeftRadii , UIntPack< _CRightRadii ... > cRightRadii , WindowSlice< RegularTreeNode* , UIntPack< ( _PLeftRadii + _PRightRadii + 1 ) ... > > pNeighbors , WindowSlice< RegularTreeNode* , UIntPack< ( _CLeftRadii + _CRightRadii + 1 ) ... > > cNeighbors , int cIdx , Allocator< RegularTreeNode >* nodeAllocator , NodeInitializer &initializer )
{
	return _NeighborsLoop< CreateNodes , ThreadSafe , NodeInitializer >( UIntPack< _PLeftRadii ... >() , UIntPack< _PRightRadii ... >() , UIntPack< _CLeftRadii ... >() , UIntPack< _CRightRadii ... >() , ( ConstWindowSlice< RegularTreeNode* , UIntPack< ( _PLeftRadii + _PRightRadii + 1 ) ... > > )pNeighbors , cNeighbors , cIdx , nodeAllocator , initializer );
}
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... LeftRadii , unsigned int ... RightRadii >
template< bool CreateNodes , bool ThreadSafe , typename NodeInitializer , unsigned int _PLeftRadius , unsigned int ... _PLeftRadii , unsigned int _PRightRadius , unsigned int ... _PRightRadii , unsigned int _CLeftRadius , unsigned int ... _CLeftRadii , unsigned int _CRightRadius , unsigned int ... _CRightRadii >
unsigned int RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::NeighborKey< UIntPack< LeftRadii ... > , UIntPack< RightRadii ... > >::_Run< CreateNodes , ThreadSafe , NodeInitializer , UIntPack< _PLeftRadius , _PLeftRadii ... > , UIntPack< _PRightRadius , _PRightRadii ... > , UIntPack< _CLeftRadius , _CLeftRadii ... > , UIntPack< _CRightRadius , _CRightRadii ... > >::Run( ConstWindowSlice< RegularTreeNode* , UIntPack< _PLeftRadius + _PRightRadius + 1 , ( _PLeftRadii + _PRightRadii + 1 ) ... > > pNeighbors , WindowSlice< RegularTreeNode* , UIntPack< _CLeftRadius + _CRightRadius + 1 , ( _CLeftRadii + _CRightRadii + 1 ) ... > > cNeighbors , int* c , int cornerIndex , Allocator< RegularTreeNode >* nodeAllocator , NodeInitializer &initializer )
{
	static const int D = sizeof ... ( _PLeftRadii ) + 1;
	unsigned int count=0;
	for( int i=-(int)_CLeftRadius ; i<=(int)_CRightRadius ; i++ )
	{
		int _i = (i+c[Dim-D]) + ( _CLeftRadius<<1 ) , pi = ( _i>>1 ) - _CLeftRadius + _PLeftRadius  , ci = i + _CLeftRadius;
		count += _Run< CreateNodes , ThreadSafe , NodeInitializer , UIntPack< _PLeftRadii ... > , UIntPack< _PRightRadii ... > , UIntPack< _CLeftRadii ... > , UIntPack< _CRightRadii ... > >::Run( pNeighbors[pi] , cNeighbors[ci] , c , cornerIndex | ( ( _i&1)<<(Dim-D) ) , nodeAllocator , initializer );
	}
	return count;
}


template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... LeftRadii , unsigned int ... RightRadii >
template< bool CreateNodes , bool ThreadSafe , typename NodeInitializer , unsigned int _PLeftRadius , unsigned int _PRightRadius , unsigned int _CLeftRadius , unsigned int _CRightRadius >
unsigned int RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::NeighborKey< UIntPack< LeftRadii ... > , UIntPack< RightRadii ... > >::_Run< CreateNodes , ThreadSafe , NodeInitializer , UIntPack< _PLeftRadius > , UIntPack< _PRightRadius > , UIntPack< _CLeftRadius > , UIntPack< _CRightRadius > >::Run( ConstWindowSlice< RegularTreeNode* , UIntPack< _PLeftRadius+_PRightRadius+1 > > pNeighbors , WindowSlice< RegularTreeNode* , UIntPack< _CLeftRadius+_CRightRadius+1 > > cNeighbors , int* c , int cornerIndex , Allocator< RegularTreeNode >* nodeAllocator , NodeInitializer &initializer )
{
	static const int D = 1;
	unsigned int count=0;
	for( int i=-(int)_CLeftRadius ; i<=(int)_CRightRadius ; i++ )
	{
		int _i = (i+c[Dim-1]) + ( _CLeftRadius<<1 ) , pi = ( _i>>1 ) - _CLeftRadius + _PLeftRadius  , ci = i + _CLeftRadius;
		if( CreateNodes )
		{
			if( pNeighbors[pi] )
			{
				if( !pNeighbors[pi]->children ) pNeighbors[pi]->template initChildren< ThreadSafe >( nodeAllocator , initializer );
				cNeighbors[ci] = pNeighbors[pi]->children + ( cornerIndex | ( ( _i&1)<<(Dim-1) ) );
				count++;
			}
			else cNeighbors[ci] = NULL;
		}
		else
		{
			if( pNeighbors[pi] && pNeighbors[pi]->children ) cNeighbors[ci] = pNeighbors[pi]->children + ( cornerIndex | ( ( _i&1)<<(Dim-1) ) ) , count++;
			else cNeighbors[ci] = NULL;
		}
	}
	return count;
}

template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... LeftRadii , unsigned int ... RightRadii >
template< bool CreateNodes , bool ThreadSafe , typename NodeInitializer >
unsigned int RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::NeighborKey< UIntPack< LeftRadii ... > , UIntPack< RightRadii ... > >::getChildNeighbors( int cIdx , int d , NeighborType& cNeighbors , Allocator< RegularTreeNode >* nodeAllocator , NodeInitializer &initializer ) const
{
	NeighborType& pNeighbors = neighbors[d];
	// Check that we actually have a center node
	if( !pNeighbors.neighbors.data[ CenterIndex ] ) return 0;
	return _NeighborsLoop< CreateNodes , ThreadSafe >( UIntPack< LeftRadii ... >() , UIntPack< RightRadii ... >() , UIntPack< LeftRadii ... >() , UIntPack< RightRadii ... >() , pNeighbors.neighbors() , cNeighbors.neighbors() , cIdx , nodeAllocator , initializer );
}

template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... LeftRadii , unsigned int ... RightRadii >
template< bool CreateNodes , bool ThreadSafe , typename NodeInitializer , class Real >
unsigned int RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::NeighborKey< UIntPack< LeftRadii ... > , UIntPack< RightRadii ... > >::getChildNeighbors( Point< Real , Dim > p , int d , NeighborType& cNeighbors , Allocator< RegularTreeNode >* nodeAllocator , NodeInitializer &initializer ) const
{
	NeighborType& pNeighbors = neighbors[d];
	// Check that we actually have a center node
	if( !pNeighbors.neighbors.data[ CenterIndex ] ) return 0;
	Point< Real , Dim > c;
	Real w;
	pNeighbors.neighbors.data[ CenterIndex ]->centerAndWidth( c , w );
	return getChildNeighbors< CreateNodes , ThreadSafe >( CornerIndex( c , p ) , d , cNeighbors , nodeAllocator , initializer );
}

template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... LeftRadii , unsigned int ... RightRadii >
template< bool CreateNodes , bool ThreadSafe , typename NodeInitializer >
typename RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::template Neighbors< UIntPack< ( LeftRadii + RightRadii + 1 ) ... > >& RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::NeighborKey< UIntPack< LeftRadii ... > , UIntPack< RightRadii ... > >::getNeighbors( RegularTreeNode< Dim , NodeData , DepthAndOffsetType >* node , Allocator< RegularTreeNode >* nodeAllocator , NodeInitializer &initializer )
{
	NeighborType& neighbors = this->neighbors[node->depth()];
	// This is required in case the neighbors have been constructed between the last call to getNeighbors and this one
	if( node==neighbors.neighbors.data[ CenterIndex ] )
	{
		bool reset = false;
		for( int i=0 ; i<WindowSize< UIntPack< ( LeftRadii+RightRadii+1 ) ... > >::Size ; i++ ) if( !neighbors.neighbors.data[i] ) reset = true;
		if( reset ) neighbors.neighbors.data[ CenterIndex ] = NULL;
	}
	if( node!=neighbors.neighbors.data[ CenterIndex ] )
	{
		for( int d=node->depth()+1 ; d<=_depth && this->neighbors[d].neighbors.data[ CenterIndex ] ; d++ ) this->neighbors[d].neighbors.data[ CenterIndex ] = NULL;
		neighbors.clear();
		if( !node->parent ) neighbors.neighbors.data[ CenterIndex ] = node;
		else _NeighborsLoop< CreateNodes , ThreadSafe >( UIntPack< LeftRadii ... >() , UIntPack< RightRadii ... >() , UIntPack< LeftRadii ... >() , UIntPack< RightRadii ... >() , getNeighbors< CreateNodes , ThreadSafe >( node->parent , nodeAllocator , initializer ).neighbors() , neighbors.neighbors() , (int)( node - node->parent->children ) , nodeAllocator , initializer );
	}
	return neighbors;
}

template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... LeftRadii , unsigned int ... RightRadii >
template< bool CreateNodes , bool ThreadSafe , typename NodeInitializer , unsigned int ... _LeftRadii , unsigned int ... _RightRadii >
void RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::NeighborKey< UIntPack< LeftRadii ... > , UIntPack< RightRadii ... > >::getNeighbors( UIntPack< _LeftRadii ... > , UIntPack< _RightRadii ... > , RegularTreeNode< Dim , NodeData , DepthAndOffsetType >* node , Neighbors< UIntPack< ( _LeftRadii + _RightRadii + 1 ) ... > >& neighbors , Allocator< RegularTreeNode >* nodeAllocator , NodeInitializer &initializer )
{
	static const unsigned int _CenterIndex = WindowIndex< UIntPack< ( _LeftRadii + _RightRadii + 1 ) ... > , UIntPack< _LeftRadii ... > >::Index;
	neighbors.clear();
	if( !node ) return;

	// [WARNING] This estimate of the required radius is somewhat conservative if the readius is odd (depending on where the node is relative to its parent)
	UIntPack<  LeftRadii ... >  leftRadii;
	UIntPack< RightRadii ... > rightRadii;
	UIntPack< (  _LeftRadii+1 )/2 ... >  pLeftRadii;
	UIntPack< ( _RightRadii+1 )/2 ... > pRightRadii;
	UIntPack<  _LeftRadii ... >  cLeftRadii;
	UIntPack< _RightRadii ... > cRightRadii;

	// If we are at the root of the tree, we are done
	if( !node->parent ) neighbors.neighbors.data[ _CenterIndex ] = node;
	// If we can get the data from the the key for the parent node, do that
	else if( pLeftRadii<=leftRadii && pRightRadii<=rightRadii )
	{
		getNeighbors< CreateNodes , ThreadSafe >( node->parent , nodeAllocator , initializer );
		const Neighbors< UIntPack< ( LeftRadii + RightRadii + 1 ) ... > >& pNeighbors = this->neighbors[ node->depth()-1 ];
		_NeighborsLoop< CreateNodes , ThreadSafe >( leftRadii , rightRadii , cLeftRadii , cRightRadii , pNeighbors.neighbors() , neighbors.neighbors() , (int)( node - node->parent->children ) , nodeAllocator , initializer );
	}
	// Otherwise recurse
	else
	{
		Neighbors< UIntPack< ( ( _LeftRadii+1 )/2  + ( _RightRadii+1 )/2 + 1 ) ... > > pNeighbors;
		getNeighbors< CreateNodes , ThreadSafe >( pLeftRadii , pRightRadii , node->parent , pNeighbors , nodeAllocator , initializer );
		_NeighborsLoop< CreateNodes , ThreadSafe >( pLeftRadii , pRightRadii , cLeftRadii , cRightRadii , pNeighbors.neighbors() , neighbors.neighbors() , (int)( node - node->parent->children ) , nodeAllocator , initializer );
	}
}

template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... LeftRadii , unsigned int ... RightRadii >
template< bool CreateNodes , bool ThreadSafe , typename NodeInitializer , unsigned int ... _LeftRadii , unsigned int ... _RightRadii >
void RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::NeighborKey< UIntPack< LeftRadii ... > , UIntPack< RightRadii ... > >::getNeighbors( UIntPack< _LeftRadii ... > , UIntPack< _RightRadii ... > , RegularTreeNode< Dim , NodeData , DepthAndOffsetType >* node , Neighbors< UIntPack< ( _LeftRadii + _RightRadii + 1 ) ... > >& pNeighbors , Neighbors< UIntPack< ( _LeftRadii + _RightRadii + 1 ) ... > >& neighbors , Allocator< RegularTreeNode >* nodeAllocator , NodeInitializer &initializer )
{
	UIntPack<  _LeftRadii ... >  leftRadii;
	UIntPack< _RightRadii ... > rightRadii;
	if( !node->parent ) getNeighbors< CreateNodes , ThreadSafe >( leftRadii , rightRadii , node , neighbors , nodeAllocator , initializer );
	else
	{
		getNeighbors< CreateNodes , ThreadSafe >( leftRadii , rightRadii , node->parent , pNeighbors , nodeAllocator , initializer );
		_NeighborsLoop< CreateNodes , ThreadSafe >( leftRadii , rightRadii , leftRadii , rightRadii , pNeighbors.neighbors() , neighbors.neighbors() , (int)( node - node->parent->children ) , nodeAllocator , initializer );
	}
}

///////////////////////////////////////
// RegularTreeNode::ConstNeighborKey //
///////////////////////////////////////
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... LeftRadii , unsigned int ... RightRadii >
RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::ConstNeighborKey< UIntPack< LeftRadii ... > , UIntPack< RightRadii ... > >::ConstNeighborKey( void ){ _depth=-1 , neighbors=NULL; }
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... LeftRadii , unsigned int ... RightRadii >
RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::ConstNeighborKey< UIntPack< LeftRadii ... > , UIntPack< RightRadii ... > >::ConstNeighborKey( const ConstNeighborKey& key )
{
	_depth = 0 , neighbors = NULL;
	set( key._depth );
	for( int d=0 ; d<=_depth ; d++ ) memcpy( &neighbors[d] , &key.neighbors[d] , sizeof( ConstNeighbors< UIntPack< ( LeftRadii + RightRadii + 1 ) ... > > ) );
}
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... LeftRadii , unsigned int ... RightRadii >
RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::ConstNeighborKey< UIntPack< LeftRadii ... > , UIntPack< RightRadii ... > >::~ConstNeighborKey( void )
{
	if( neighbors ) delete[] neighbors;
	neighbors=NULL;
}
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... LeftRadii , unsigned int ... RightRadii >
typename RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::template ConstNeighborKey< UIntPack< LeftRadii ... > , UIntPack< RightRadii ... > >& RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::ConstNeighborKey< UIntPack< LeftRadii ... > , UIntPack< RightRadii ... > >::operator = ( const ConstNeighborKey& key )
{
	set( key._depth );
	for( int d=0 ; d<=_depth ; d++ ) memcpy( &neighbors[d] , &key.neighbors[d] , sizeof( ConstNeighbors< UIntPack< ( LeftRadii + RightRadii + 1 ) ... > > ) );
}
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... LeftRadii , unsigned int ... RightRadii >
void RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::ConstNeighborKey< UIntPack< LeftRadii ... > , UIntPack< RightRadii ... > >::set( int d )
{
	if( neighbors ) delete[] neighbors;
	neighbors = NULL;
	_depth = d;
	if( d<0 ) return;
	neighbors = new NeighborType[d+1];
}

template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... LeftRadii , unsigned int ... RightRadii >
template< unsigned int ... _PLeftRadii , unsigned int ... _PRightRadii , unsigned int ... _CLeftRadii , unsigned int ... _CRightRadii >
unsigned int RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::ConstNeighborKey< UIntPack< LeftRadii ... > , UIntPack< RightRadii ... > >::_NeighborsLoop( UIntPack< _PLeftRadii ... > pLeftRadii , UIntPack< _PRightRadii ... > pRightRadii , UIntPack< _CLeftRadii ... > cLeftRadii , UIntPack< _CRightRadii ... > cRightRadii , ConstWindowSlice< const RegularTreeNode * , UIntPack< ( _PLeftRadii + _PRightRadii + 1 ) ... > > pNeighbors , WindowSlice< const RegularTreeNode * , UIntPack< ( _CLeftRadii + _CRightRadii + 1 ) ... > > cNeighbors , int cIdx )
{
	static_assert( Dim==sizeof ... ( _PLeftRadii ) && Dim==sizeof ... ( _PRightRadii ) && Dim==sizeof ... ( _CLeftRadii ) && Dim==sizeof ... ( _CRightRadii ) , "[ERROR] Dimensions don't match" );
	int c[Dim];
	for( int d=0 ; d<Dim ; d++ ) c[d] = ( cIdx>>d ) & 1;
	return _Run< UIntPack< _PLeftRadii ... > , UIntPack< _PRightRadii ... > , UIntPack< _CLeftRadii ... > , UIntPack< _CRightRadii ... > >::Run( pNeighbors , cNeighbors , c , 0 );
}
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... LeftRadii , unsigned int ... RightRadii >
template< unsigned int ... _PLeftRadii , unsigned int ... _PRightRadii , unsigned int ... _CLeftRadii , unsigned int ... _CRightRadii >
unsigned int RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::ConstNeighborKey< UIntPack< LeftRadii ... > , UIntPack< RightRadii ... > >::_NeighborsLoop( UIntPack< _PLeftRadii ... > pLeftRadii , UIntPack< _PRightRadii ... > pRightRadii , UIntPack< _CLeftRadii ... > cLeftRadii , UIntPack< _CRightRadii ... > cRightRadii , WindowSlice< const RegularTreeNode* , UIntPack< ( _PLeftRadii + _PRightRadii + 1 ) ... > > pNeighbors , WindowSlice< const RegularTreeNode* , UIntPack< ( _CLeftRadii + _CRightRadii + 1 ) ... > > cNeighbors , int cIdx )
{
	return _NeighborsLoop( UIntPack< _PLeftRadii ... >() , UIntPack< _PRightRadii ... >() , UIntPack< _CLeftRadii ... >() , UIntPack< _CRightRadii ... >() , ( ConstWindowSlice< const RegularTreeNode* , UIntPack< ( _PLeftRadii + _PRightRadii + 1 ) ... > > )pNeighbors , cNeighbors , cIdx );
}

template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... LeftRadii , unsigned int ... RightRadii >
template< unsigned int _PLeftRadius , unsigned int ... _PLeftRadii , unsigned int _PRightRadius , unsigned int ... _PRightRadii , unsigned int _CLeftRadius , unsigned int ... _CLeftRadii , unsigned int _CRightRadius , unsigned int ... _CRightRadii >
unsigned int RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::ConstNeighborKey< UIntPack< LeftRadii ... > , UIntPack< RightRadii ... > >::_Run< UIntPack< _PLeftRadius , _PLeftRadii ... > , UIntPack< _PRightRadius , _PRightRadii ... > , UIntPack< _CLeftRadius , _CLeftRadii ... > , UIntPack< _CRightRadius , _CRightRadii ... > >::Run( ConstWindowSlice< const RegularTreeNode* , UIntPack< _PLeftRadius + _PRightRadius + 1 , ( _PLeftRadii + _PRightRadii + 1 ) ... > > pNeighbors , WindowSlice< const RegularTreeNode* , UIntPack< _CLeftRadius + _CRightRadius + 1 , ( _CLeftRadii + _CRightRadii + 1 ) ... > > cNeighbors , int* c , int cornerIndex )
{
	static const int D = sizeof ... ( _PLeftRadii ) + 1;
	unsigned int count=0;
	for( int i=-(int)_CLeftRadius ; i<=(int)_CRightRadius ; i++ )
	{
		int _i = (i+c[Dim-D]) + ( _CLeftRadius<<1 ) , pi = ( _i>>1 ) - _CLeftRadius + _PLeftRadius  , ci = i + _CLeftRadius;
		count += _Run< UIntPack< _PLeftRadii ... > , UIntPack< _PRightRadii ... > , UIntPack< _CLeftRadii ... > , UIntPack<  _CRightRadii ... > >::Run( pNeighbors[pi] , cNeighbors[ci] , c , cornerIndex | ( ( _i&1)<<(Dim-D) ) );
	}
	return count;
}
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... LeftRadii , unsigned int ... RightRadii >
template< unsigned int _PLeftRadius , unsigned int _PRightRadius , unsigned int _CLeftRadius , unsigned int _CRightRadius  >
unsigned int RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::ConstNeighborKey< UIntPack< LeftRadii ... > , UIntPack< RightRadii ... > >::_Run< UIntPack< _PLeftRadius > , UIntPack< _PRightRadius > , UIntPack< _CLeftRadius > , UIntPack< _CRightRadius > >::Run( ConstWindowSlice< const RegularTreeNode* , UIntPack< _PLeftRadius+_PRightRadius+1 > > pNeighbors , WindowSlice< const RegularTreeNode* , UIntPack< _CLeftRadius+_CRightRadius+1 > > cNeighbors , int* c , int cornerIndex )
{
	static const int D = 1;
	unsigned int count=0;
	for( int i=-(int)_CLeftRadius ; i<=(int)_CRightRadius ; i++ )
	{
		int _i = (i+c[Dim-D]) + ( _CLeftRadius<<1 ) , pi = ( _i>>1 ) - _CLeftRadius + _PLeftRadius  , ci = i + _CLeftRadius;
		if( pNeighbors[pi] && pNeighbors[pi]->children ) cNeighbors[ci] = pNeighbors[pi]->children + ( cornerIndex | ( ( _i&1)<<(Dim-1) ) ) , count++;
		else cNeighbors[ci] = NULL;
	}
	return count;
}

template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... LeftRadii , unsigned int ... RightRadii >
unsigned int RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::ConstNeighborKey< UIntPack< LeftRadii ... > , UIntPack< RightRadii ... > >::getChildNeighbors( int cIdx , int d , ConstNeighbors< UIntPack< ( LeftRadii + RightRadii + 1 ) ... > >& cNeighbors ) const
{
	const ConstNeighbors< UIntPack< ( LeftRadii + RightRadii + 1 ) ... > >& pNeighbors = neighbors[d];
	// Check that we actually have a center node
	if( !pNeighbors.neighbors.data[ CenterIndex ] ) return 0;

	return _NeighborsLoop( UIntPack< LeftRadii ... >() , UIntPack< RightRadii ... >() , UIntPack< LeftRadii ... >() , UIntPack< RightRadii ... >() , pNeighbors.neighbors() , cNeighbors.neighbors() , cIdx );
}

template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... LeftRadii , unsigned int ... RightRadii >
typename RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::template ConstNeighbors< UIntPack< ( LeftRadii + RightRadii + 1 ) ... > >& RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::ConstNeighborKey< UIntPack< LeftRadii ... > , UIntPack< RightRadii ... > >::getNeighbors( const RegularTreeNode* node )
{
	ConstNeighbors< UIntPack< ( LeftRadii + RightRadii + 1 ) ... > >& neighbors = this->neighbors[ node->depth() ];
	if( node!=neighbors.neighbors.data[ CenterIndex ] )
	{
		for( int d=node->depth()+1 ; d<=_depth && this->neighbors[d].neighbors.data[ CenterIndex ] ; d++ ) this->neighbors[d].neighbors.data[ CenterIndex ] = NULL;
		neighbors.clear();
		if( !node->parent ) neighbors.neighbors.data[ CenterIndex ] = node;
		else _NeighborsLoop( UIntPack< LeftRadii ... >() , UIntPack< RightRadii ... >() , UIntPack< LeftRadii ... >() , UIntPack< RightRadii ... >() , getNeighbors( node->parent ).neighbors() , neighbors.neighbors() , (int)( node - node->parent->children ) );
	}
	return neighbors;
}
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... LeftRadii , unsigned int ... RightRadii >
template< unsigned int ... _LeftRadii , unsigned int ... _RightRadii >
void RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::ConstNeighborKey< UIntPack< LeftRadii ... > , UIntPack< RightRadii ... > >::getNeighbors( UIntPack< _LeftRadii ... > , UIntPack< _RightRadii ... > , const RegularTreeNode< Dim , NodeData , DepthAndOffsetType >* node , ConstNeighbors< UIntPack< ( _LeftRadii + _RightRadii + 1 ) ... > >& neighbors )
{
	static const unsigned int _CenterIndex = WindowIndex< UIntPack< ( _LeftRadii + _RightRadii + 1 ) ... > , UIntPack< _LeftRadii ... > >::Index;

	neighbors.clear();
	if( !node ) return;

	UIntPack<  LeftRadii ... >  leftRadii;
	UIntPack< RightRadii ... > rightRadii;
	UIntPack< (  _LeftRadii+1 )/2 ... >  pLeftRadii;
	UIntPack< ( _RightRadii+1 )/2 ... > pRightRadii;
	UIntPack<  _LeftRadii ... >  cLeftRadii;
	UIntPack< _RightRadii ... > cRightRadii;
	// If we are at the root of the tree, we are done
	if( !node->parent ) neighbors.neighbors.data[ _CenterIndex ] = node;
	// If we can get the data from the the key for the parent node, do that
	else if( pLeftRadii<=leftRadii && pRightRadii<=rightRadii )
	{
		getNeighbors( node->parent );
		const ConstNeighbors< UIntPack< ( LeftRadii + RightRadii + 1 ) ... > >& pNeighbors = this->neighbors[ node->depth()-1 ];
		_NeighborsLoop( leftRadii , rightRadii , cLeftRadii , cRightRadii , pNeighbors.neighbors() , neighbors.neighbors() , (int)( node - node->parent->children ) );
	}
	// Otherwise recurse
	else
	{
		ConstNeighbors< UIntPack< ( ( _LeftRadii+1 )/2  + ( _RightRadii+1 )/2 + 1 ) ... > > pNeighbors;
		getNeighbors( pLeftRadii , pRightRadii , node->parent , pNeighbors );
		_NeighborsLoop( pLeftRadii , pRightRadii , cLeftRadii , cRightRadii , pNeighbors.neighbors() , neighbors.neighbors() , (int)( node - node->parent->children ) );
	}
	return;
}
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... LeftRadii , unsigned int ... RightRadii >
template< unsigned int ... _LeftRadii , unsigned int ... _RightRadii >
void RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::ConstNeighborKey< UIntPack< LeftRadii ... > , UIntPack< RightRadii ... > >::getNeighbors( UIntPack< _LeftRadii ... > , UIntPack< _RightRadii ... > , const RegularTreeNode< Dim , NodeData , DepthAndOffsetType >* node , ConstNeighbors< UIntPack< ( _LeftRadii + _RightRadii + 1 ) ... > >& pNeighbors , ConstNeighbors< UIntPack< ( _LeftRadii + _RightRadii + 1 ) ... > >& neighbors )
{
	UIntPack<  _LeftRadii ... >  leftRadii;
	UIntPack< _RightRadii ... > rightRadii;
	if( !node->parent ) return getNeighbors( leftRadii , rightRadii , node , neighbors );
	else
	{
		 getNeighbors( leftRadii , rightRadii , node->parent , pNeighbors );
		_NeighborsLoop( leftRadii , rightRadii , leftRadii , rightRadii , pNeighbors.neighbors() , neighbors.neighbors() , (int)( node - node->parent->children ) );
	}
}
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... LeftRadii , unsigned int ... RightRadii >
template< class Real >
unsigned int RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::ConstNeighborKey< UIntPack< LeftRadii ... > , UIntPack< RightRadii ... > >::getChildNeighbors( Point< Real , Dim > p , int d , ConstNeighbors< UIntPack< ( LeftRadii + RightRadii + 1 ) ... > >& cNeighbors ) const
{
	ConstNeighbors< UIntPack< ( LeftRadii + RightRadii + 1 ) ... > >& pNeighbors = neighbors[d];
	// Check that we actually have a center node
	if( !pNeighbors.neighbors.data[ CenterIndex ] ) return 0;
	Point< Real , Dim > c;
	Real w;
	pNeighbors.neighbors.data[ CenterIndex ]->centerAndWidth( c , w );
	int cIdx = 0;
	for( int d=0 ; d<Dim ; d++ ) if( p[d]>c[d] ) cIdx |= (1<<d);
	return getChildNeighbors( cIdx , d , cNeighbors );
}
