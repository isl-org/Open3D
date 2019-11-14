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

/////////////////////
// SortedTreeNodes //
/////////////////////
template< unsigned int Dim >
SortedTreeNodes< Dim >::SortedTreeNodes( void )
{
	_sliceStart = NullPointer( Pointer( node_index_type ) );
	treeNodes = NullPointer( TreeNode* );
	_levels = 0;
}
template< unsigned int Dim >
SortedTreeNodes< Dim >::~SortedTreeNodes( void )
{
	if( _sliceStart ) for( int d=0 ; d<_levels ; d++ ) FreePointer( _sliceStart[d] );
	FreePointer( _sliceStart );
	DeletePointer( treeNodes );
}
template< unsigned int Dim >
void SortedTreeNodes< Dim >::set( TreeNode& root , std::vector< node_index_type >* map )
{
	size_t sz = set( root );

	if( map )
	{
		map->resize( sz , -1 );
		for( node_index_type i=0 ; i<_sliceStart[_levels-1][(size_t)1<<(_levels-1)] ; i++ ) if( treeNodes[i]->nodeData.nodeIndex>=0 ) (*map)[ treeNodes[i]->nodeData.nodeIndex ] = i;
	}
	for( node_index_type i=0 ; i<_sliceStart[_levels-1][(size_t)1<<(_levels-1)] ; i++ ) treeNodes[i]->nodeData.nodeIndex = i;
}
template< unsigned int Dim >
size_t SortedTreeNodes< Dim >::set( TreeNode& root )
{
	size_t sz = 0;
	_levels = root.maxDepth()+1;

	if( _sliceStart ) for( int d=0 ; d<_levels ; d++ ) FreePointer( _sliceStart[d] );
	FreePointer( _sliceStart );
	DeletePointer( treeNodes );

	_sliceStart = AllocPointer< Pointer( node_index_type ) >( _levels );
	for( int l=0 ; l<_levels ; l++ )
	{
		_sliceStart[l] = AllocPointer< node_index_type >( ((size_t)1<<l)+1 );
		memset( _sliceStart[l] , 0 , sizeof(node_index_type)*( ((size_t)1<<l)+1 ) );
	}

	// Count the number of nodes in each slice
	for( TreeNode* node = root.nextNode() ; node ; node = root.nextNode( node ) )
	{
		if( node->nodeData.nodeIndex>=0 ) sz = std::max< size_t >( node->nodeData.nodeIndex+1 , sz );
		if( !GetGhostFlag< Dim >( node ) )
		{
			int d , off[Dim];
			node->depthAndOffset( d , off );
			_sliceStart[d][ off[Dim-1]+1 ]++;
		}
	}

	// Get the start index for each slice
	{
		node_index_type levelOffset = 0;
		for( int l=0 ; l<_levels ; l++ )
		{
			_sliceStart[l][0] = levelOffset;
			for( int s=0 ; s<((size_t)1<<l); s++ ) _sliceStart[l][s+1] += _sliceStart[l][s];
			levelOffset = _sliceStart[l][(size_t)1<<l];
		}
	}
	// Allocate memory for the tree nodes
	treeNodes = NewPointer< TreeNode* >( _sliceStart[_levels-1][(size_t)1<<(_levels-1)] );

	// Add the tree nodes
	for( TreeNode* node=root.nextNode() ; node ; node=root.nextNode( node ) ) if( !GetGhostFlag< Dim >( node ) )
	{
		int d , off[Dim];
		node->depthAndOffset( d , off );
		treeNodes[ _sliceStart[d][ off[Dim-1] ]++ ] = node;
	}

	// Shift the slice offsets up since we incremented as we added
	for( int l=0 ; l<_levels ; l++ )
	{
		for( int s=(1<<l) ; s>0 ; s-- ) _sliceStart[l][s] = _sliceStart[l][s-1];
		_sliceStart[l][0] = l>0 ? _sliceStart[l-1][(size_t)1<<(l-1)] : 0;
	}
	return sz;
}

