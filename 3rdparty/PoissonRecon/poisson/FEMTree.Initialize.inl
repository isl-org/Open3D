/*
Copyright (c) 2016, Michael Kazhdan
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

////////////////////////
// FEMTreeInitializer //
////////////////////////
template< unsigned int Dim , class Real >
size_t FEMTreeInitializer< Dim , Real >::Initialize( FEMTreeNode& node , int maxDepth , std::function< bool ( int , int[] ) > Refine , Allocator< FEMTreeNode >* nodeAllocator , std::function< void ( FEMTreeNode& ) > NodeInitializer )
{
	size_t count = 0;
	int d , off[3];
	node.depthAndOffset( d , off );
	if( node.depth()<maxDepth && Refine( d , off ) )
	{
		node.initChildren< false >( nodeAllocator , NodeInitializer ) , count += 1<<Dim;
		for( int c=0 ; c<(1<<Dim) ; c++ ) count += Initialize( node.children[c] , maxDepth , Refine , nodeAllocator , NodeInitializer );
	}
	return count;
}

template< unsigned int Dim , class Real >
size_t FEMTreeInitializer< Dim , Real >::Initialize( FEMTreeNode& root , InputPointStream< Real , Dim >& pointStream , int maxDepth , std::vector< PointSample >& samplePoints , Allocator< FEMTreeNode >* nodeAllocator , std::function< void ( FEMTreeNode& ) > NodeInitializer )
{
	auto Leaf = [&]( FEMTreeNode& root , Point< Real , Dim > p , int maxDepth )
	{
		for( int d=0 ; d<Dim ; d++ ) if( p[d]<0 || p[d]>1 ) return (FEMTreeNode*)NULL;
		Point< Real , Dim > center;
		for( int d=0 ; d<Dim ; d++ ) center[d] = (Real)0.5;
		Real width = Real(1.0);
		FEMTreeNode* node = &root;
		int d = 0;
		while( d<maxDepth )
		{
			if( !node->children ) node->template initChildren< false >( nodeAllocator , NodeInitializer );
			int cIndex = FEMTreeNode::ChildIndex( center , p );
			node = node->children + cIndex;
			d++;
			width /= 2;
			for( int dd=0 ; dd<Dim ; dd++ )
				if( (cIndex>>dd) & 1 ) center[dd] += width/2;
				else                   center[dd] -= width/2;
		}
		return node;
	};

	// Add the point data
	size_t outOfBoundPoints = 0 , pointCount = 0;
	{
		std::vector< node_index_type > nodeToIndexMap;
		Point< Real , Dim > p;
		while( pointStream.nextPoint( p ) )
		{
			Real weight = (Real)1.;
			FEMTreeNode* temp = Leaf( root , p , maxDepth );
			if( !temp ){ outOfBoundPoints++ ; continue; }
			node_index_type nodeIndex = temp->nodeData.nodeIndex;
			if( nodeIndex>=(node_index_type)nodeToIndexMap.size() ) nodeToIndexMap.resize( nodeIndex+1 , -1 );
			node_index_type idx = nodeToIndexMap[ nodeIndex ];
			if( idx==-1 )
			{
				idx = (node_index_type)samplePoints.size();
				nodeToIndexMap[ nodeIndex ] = idx;
				samplePoints.resize( idx+1 ) , samplePoints[idx].node = temp;
			}
			samplePoints[idx].sample += ProjectiveData< Point< Real , Dim > , Real >( p*weight , weight );
			pointCount++;
		}
		pointStream.reset();
	}
	if( outOfBoundPoints  ) WARN( "Found out-of-bound points: " , outOfBoundPoints );
	if( std::is_same< Real , float >::value )
	{
		std::vector< size_t > badNodeCounts( ThreadPool::NumThreads() , 0 );
		ThreadPool::Parallel_for( 0 , samplePoints.size() , [&]( unsigned int thread , size_t i )
		{
			Point< Real , Dim > start;
			Real width;
			samplePoints[i].node->startAndWidth( start , width );
			Point< Real , Dim > p = samplePoints[i].sample.data / samplePoints[i].sample.weight;
			bool foundBadNode = false;
			for( int d=0 ; d<Dim ; d++ )
			{
				if     ( p[d]<start[d]       ) foundBadNode = true , p[d] = start[d];
				else if( p[d]>start[d]+width ) foundBadNode = true , p[d] = start[d] + width;
			}
			if( foundBadNode )
			{
				samplePoints[i].sample.data = p * samplePoints[i].sample.weight;
				badNodeCounts[ thread ]++;
			}
		}
		);
		size_t badNodeCount = 0;
		for( int i=0 ; i<badNodeCounts.size() ; i++ ) badNodeCount += badNodeCounts[i];
		if( badNodeCount ) WARN( "Found bad sample nodes: " , badNodeCount );
	}
	FEMTree< Dim , Real >::MemoryUsage();
	return pointCount;
}

template< unsigned int Dim , class Real >
template< class Data >
size_t FEMTreeInitializer< Dim , Real >::Initialize( FEMTreeNode& root , InputPointStreamWithData< Real , Dim , Data >& pointStream , int maxDepth , std::vector< PointSample >& samplePoints , std::vector< Data >& sampleData , bool mergeNodeSamples , Allocator< FEMTreeNode >* nodeAllocator , std::function< void ( FEMTreeNode& ) > NodeInitializer , std::function< Real ( const Point< Real , Dim >& , Data& ) > ProcessData )
{
	auto Leaf = [&]( FEMTreeNode& root , Point< Real , Dim > p , int maxDepth )
	{
		for( int d=0 ; d<Dim ; d++ ) if( p[d]<0 || p[d]>1 ) return (FEMTreeNode*)NULL;
		Point< Real , Dim > center;
		for( int d=0 ; d<Dim ; d++ ) center[d] = (Real)0.5;
		Real width = Real(1.0);
		FEMTreeNode* node = &root;
		int d = 0;
		while( d<maxDepth )
		{
			if( !node->children ) node->template initChildren< false >( nodeAllocator , NodeInitializer );
			int cIndex = FEMTreeNode::ChildIndex( center , p );
			node = node->children + cIndex;
			d++;
			width /= 2;
			for( int dd=0 ; dd<Dim ; dd++ )
				if( (cIndex>>dd) & 1 ) center[dd] += width/2;
				else                   center[dd] -= width/2;
		}
		return node;
	};

	// Add the point data
	size_t outOfBoundPoints = 0 , badData = 0 , pointCount = 0;
	{
		std::vector< node_index_type > nodeToIndexMap;
		Point< Real , Dim > p;
		Data d;

		while( pointStream.nextPoint( p , d ) )
		{
			Real weight = ProcessData( p , d );
			if( weight<=0 ){ badData++ ; continue; }
			FEMTreeNode* temp = Leaf( root , p , maxDepth );
			if( !temp ){ outOfBoundPoints++ ; continue; }
			node_index_type nodeIndex = temp->nodeData.nodeIndex;
			if( mergeNodeSamples )
			{
				if( nodeIndex>=(node_index_type)nodeToIndexMap.size() ) nodeToIndexMap.resize( nodeIndex+1 , -1 );
				node_index_type idx = nodeToIndexMap[ nodeIndex ];
				if( idx==-1 )
				{
					idx = (node_index_type)samplePoints.size();
					nodeToIndexMap[ nodeIndex ] = idx;
					samplePoints.resize( idx+1 ) , samplePoints[idx].node = temp;
					sampleData.resize( idx+1 );
				}
				samplePoints[idx].sample += ProjectiveData< Point< Real , Dim > , Real >( p*weight , weight );
				sampleData[ idx ] += d*weight;
			}
			else
			{
				node_index_type idx = (node_index_type)samplePoints.size();
				samplePoints.resize( idx+1 ) , sampleData.resize( idx+1 );
				samplePoints[idx].node = temp;
				samplePoints[idx].sample = ProjectiveData< Point< Real , Dim > , Real >( p*weight , weight );
				sampleData[ idx ] = d*weight;
			}
			pointCount++;
		}
		pointStream.reset();
	}
	if( outOfBoundPoints  ) WARN( "Found out-of-bound points: " , outOfBoundPoints );
	if( badData           ) WARN( "Found bad data: " , badData );
	if( std::is_same< Real , float >::value )
	{
		std::vector< size_t > badNodeCounts( ThreadPool::NumThreads() , 0 );
		ThreadPool::Parallel_for( 0 , samplePoints.size() , [&]( unsigned int thread , size_t i )
		{
			Point< Real , Dim > start;
			Real width;
			samplePoints[i].node->startAndWidth( start , width );
			Point< Real , Dim > p = samplePoints[i].sample.data / samplePoints[i].sample.weight;
			bool foundBadNode = false;
			for( int d=0 ; d<Dim ; d++ )
			{
				if     ( p[d]<start[d]       ) foundBadNode = true , p[d] = start[d];
				else if( p[d]>start[d]+width ) foundBadNode = true , p[d] = start[d] + width;
			}
			if( foundBadNode )
			{
				samplePoints[i].sample.data = p * samplePoints[i].sample.weight;
				badNodeCounts[ thread ]++;
			}
		}
		);
		size_t badNodeCount = 0;
		for( int i=0 ; i<badNodeCounts.size() ; i++ ) badNodeCount += badNodeCounts[i];
		if( badNodeCount ) WARN( "Found bad sample nodes: " , badNodeCount );
	}
	FEMTree< Dim , Real >::MemoryUsage();
	return pointCount;
}
template< unsigned int Dim , class Real >
void FEMTreeInitializer< Dim , Real >::Initialize( FEMTreeNode& root , const std::vector< Point< Real , Dim > >& vertices , const std::vector< SimplexIndex< Dim-1 , node_index_type > >& simplices , int maxDepth , std::vector< PointSample >& samples , bool mergeNodeSamples , std::vector< Allocator< FEMTreeNode > * > &nodeAllocators , std::function< void ( FEMTreeNode& ) > NodeInitializer )
{
	std::vector< node_index_type > nodeToIndexMap;
	ThreadPool::Parallel_for( 0 , simplices.size() , [&]( unsigned int t , size_t  i )
	{
		Simplex< Real , Dim , Dim-1 > s;
		for( int k=0 ; k<Dim ; k++ ) s[k] = vertices[ simplices[i][k] ];
		if( mergeNodeSamples ) _AddSimplex< true >( root , s , maxDepth , samples , &nodeToIndexMap , nodeAllocators.size() ? nodeAllocators[t] : NULL , NodeInitializer );
		else                   _AddSimplex< true >( root , s , maxDepth , samples , NULL ,            nodeAllocators.size() ? nodeAllocators[t] : NULL , NodeInitializer );
	}
	);
	FEMTree< Dim , Real >::MemoryUsage();
}

template< unsigned int Dim , class Real >
template< bool ThreadSafe >
size_t FEMTreeInitializer< Dim , Real >::_AddSimplex( FEMTreeNode& root , Simplex< Real , Dim , Dim-1 >& s , int maxDepth , std::vector< PointSample >& samples , std::vector< node_index_type >* nodeToIndexMap , Allocator< FEMTreeNode >* nodeAllocator , std::function< void ( FEMTreeNode& ) > NodeInitializer )
{
	std::vector< Simplex< Real , Dim , Dim-1 > > subSimplices;
	subSimplices.push_back( s );

	// Clip the simplex to the unit cube
	{
		for( int d=0 ; d<Dim ; d++ )
		{
			Point< Real , Dim > n;
			n[d] = 1;
			{
				std::vector< Simplex< Real , Dim , Dim-1 > > back , front;
				for( int i=0 ; i<subSimplices.size() ; i++ ) subSimplices[i].split( n , 0 , back , front );
				subSimplices = front;
			}
			{
				std::vector< Simplex< Real , Dim , Dim-1 > > back , front;
				for( int i=0 ; i<subSimplices.size() ; i++ ) subSimplices[i].split( n , 1 , back , front );
				subSimplices = back;
			}
		}
	}

	struct RegularGridIndex
	{
		int idx[Dim];
		bool operator != ( const RegularGridIndex& i ) const
		{
			for( int d=0 ; d<Dim ; d++ ) if( idx[d]!=i.idx[d] ) return true;
			return false;
		}
	};

	auto Leaf = [&]( Point< Real , Dim > p , int maxDepth )
	{
		for( int d=0 ; d<Dim ; d++ ) if( p[d]<0 || p[d]>1 ) return (FEMTreeNode*)NULL;
		Point< Real , Dim > center;
		for( int d=0 ; d<Dim ; d++ ) center[d] = (Real)0.5;
		Real width = Real(1.0);
		FEMTreeNode* node = &root;
		int d=0;
		while( d<maxDepth )
		{
			if( !node->children ) node->template initChildren< ThreadSafe >( nodeAllocator , NodeInitializer );
			int cIndex = FEMTreeNode::ChildIndex( center , p );
			node = node->children + cIndex;
			d++;
			width /= 2;
			for( int d=0 ; d<Dim ; d++ )
				if( (cIndex>>d) & 1 ) center[d] += width/2;
				else                  center[d] -= width/2;
		}
		return node;
	};

	size_t sCount = 0;
	for( int i=0 ; i<subSimplices.size() ; i++ )
	{
		// Find the finest depth at which the simplex is entirely within a node
		int tDepth;
		RegularGridIndex idx0 , idx;
		for( tDepth=0 ; tDepth<maxDepth ; tDepth++ )
		{
			// Get the grid index of the first vertex of the simplex
			for( int d=0 ; d<Dim ; d++ ) idx0.idx[d] = idx.idx[d] = (int)( subSimplices[i][0][d] * (1<<(tDepth+1)) );
			bool done = false;
			for( int k=0 ; k<Dim && !done ; k++ )
			{
				for( int d=0 ; d<Dim ; d++ ) idx.idx[d] = (int)( subSimplices[i][k][d] * (1<<(tDepth+1)) );
				if( idx!=idx0 ) done = true;
			}
			if( done ) break;
		}

		// Generate a point in the middle of the simplex
		for( int i=0 ; i<subSimplices.size() ; i++ ) sCount += _AddSimplex< ThreadSafe >( Leaf( subSimplices[i].center() , tDepth ) , subSimplices[i] , maxDepth , samples , nodeToIndexMap , nodeAllocator , NodeInitializer );
	}
	return sCount;
}

template< unsigned int Dim , class Real >
template< bool ThreadSafe >
size_t FEMTreeInitializer< Dim , Real >::_AddSimplex( FEMTreeNode* node , Simplex< Real , Dim , Dim-1 >& s , int maxDepth , std::vector< PointSample >& samples , std::vector< node_index_type >* nodeToIndexMap , Allocator< FEMTreeNode >* nodeAllocator , std::function< void ( FEMTreeNode& ) > NodeInitializer )
{
	int d = node->depth();
	if( d==maxDepth )
	{
		Real weight = s.measure();
		Point< Real , Dim > position = s.center() , normal;
		{
			Point< Real , Dim > v[Dim-1];
			for( int k=0 ; k<Dim-1 ; k++ ) v[k] = s[k+1]-s[0];
			normal = Point< Real , Dim >::CrossProduct( v );
		}
		if( weight && weight==weight )
		{
			if( nodeToIndexMap )
			{
				node_index_type nodeIndex = node->nodeData.nodeIndex;
				{
					static std::mutex m;
					std::lock_guard< std::mutex > lock(m);
					if( nodeIndex>=(node_index_type)nodeToIndexMap->size() ) nodeToIndexMap->resize( nodeIndex+1 , -1 );
					node_index_type idx = (*nodeToIndexMap)[ nodeIndex ];
					if( idx==-1 )
					{
						idx = (node_index_type)samples.size();
						(*nodeToIndexMap)[ nodeIndex ] = idx;
						samples.resize( idx+1 );
						samples[idx].node = node;
					}
					samples[idx].sample += ProjectiveData< Point< Real , Dim > , Real >( position*weight , weight );
				}
			}
			else
			{
				{
					static std::mutex m;
					std::lock_guard< std::mutex > lock(m);
					node_index_type idx = (node_index_type)samples.size();
					samples.resize( idx+1 );
					samples[idx].node = node;
					samples[idx].sample = ProjectiveData< Point< Real , Dim > , Real >( position*weight , weight );
				}
			}
		}
		return 1;
	}
	else
	{
		size_t sCount = 0;
		if( !node->children ) node->template initChildren< ThreadSafe >( nodeAllocator , NodeInitializer );

		// Split up the simplex and pass the parts on to the children
		Point< Real , Dim > center;
		Real width;
		node->centerAndWidth( center , width );

		std::vector< std::vector< Simplex< Real , Dim , Dim-1 > > > childSimplices( 1 );
		childSimplices[0].push_back( s );
		for( int d=0 ; d<Dim ; d++ )
		{
			Point< Real , Dim > n ; n[Dim-d-1] = 1;
			std::vector< std::vector< Simplex< Real , Dim , Dim-1 > > > temp( (int)( 1<<(d+1) ) );
			for( int c=0 ; c<(1<<d) ; c++ ) for( size_t i=0 ; i<childSimplices[c].size() ; i++ ) childSimplices[c][i].split( n , center[Dim-d-1] , temp[2*c] , temp[2*c+1] );
			childSimplices = temp;
		}
		for( int c=0 ; c<(1<<Dim) ; c++ ) for( size_t i=0 ; i<childSimplices[c].size() ; i++ ) if( childSimplices[c][i].measure() ) sCount += _AddSimplex< ThreadSafe >( node->children+c , childSimplices[c][i] , maxDepth , samples , nodeToIndexMap , nodeAllocator , NodeInitializer );
		return sCount;
	}
}

template< unsigned int Dim , class Real >
void FEMTreeInitializer< Dim , Real >::Initialize( FEMTreeNode& root , const std::vector< Point< Real , Dim > >& vertices , const std::vector< SimplexIndex< Dim-1 , node_index_type > >& simplices , int maxDepth , std::vector< NodeSimplices< Dim , Real > >& nodeSimplices , Allocator< FEMTreeNode > *nodeAllocator , std::function< void ( FEMTreeNode& ) > NodeInitializer )
{
	std::vector< size_t > nodeToIndexMap;
	for( size_t i=0 ; i<simplices.size() ; i++ )
	{
		Simplex< Real , Dim , Dim-1 > s;
		for( int k=0 ; k<Dim ; k++ ) s[k] = vertices[ simplices[i][k] ];
		_AddSimplex< false >( root , s , maxDepth , nodeSimplices , nodeToIndexMap , nodeAllocator , NodeInitializer );
	}
	FEMTree< Dim , Real >::MemoryUsage();
}

template< unsigned int Dim , class Real >
template< bool ThreadSafe >
size_t FEMTreeInitializer< Dim , Real >::_AddSimplex( FEMTreeNode& root , Simplex< Real , Dim , Dim-1 >& s , int maxDepth , std::vector< NodeSimplices< Dim , Real > >& simplices , std::vector< node_index_type >& nodeToIndexMap , Allocator< FEMTreeNode >* nodeAllocator , std::function< void ( FEMTreeNode& ) > NodeInitializer )
{
	std::vector< Simplex< Real , Dim , Dim-1 > > subSimplices;
	subSimplices.push_back( s );

	// Clip the simplex to the unit cube
	{
		for( int d=0 ; d<Dim ; d++ )
		{
			Point< Real , Dim > n;
			n[d] = 1;
			{
				std::vector< Simplex< Real , Dim , Dim-1 > > back , front;
				for( size_t i=0 ; i<subSimplices.size() ; i++ ) subSimplices[i].split( n , 0 , back , front );
				subSimplices = front;
			}
			{
				std::vector< Simplex< Real , Dim , Dim-1 > > back , front;
				for( size_t i=0 ; i<subSimplices.size() ; i++ ) subSimplices[i].split( n , 1 , back , front );
				subSimplices = back;
			}
		}
	}

	struct RegularGridIndex
	{
		int idx[Dim];
		bool operator != ( const RegularGridIndex& i ) const
		{
			for( int d=0 ; d<Dim ; d++ ) if( idx[d]!=i.idx[d] ) return true;
			return false;
		}
	};

	auto Leaf = [&]( Point< Real , Dim > p , int maxDepth )
	{
		for( int d=0 ; d<Dim ; d++ ) if( p[d]<0 || p[d]>1 ) return (FEMTreeNode*)NULL;
		Point< Real , Dim > center;
		for( int d=0 ; d<Dim ; d++ ) center[d] = (Real)0.5;
		Real width = Real(1.0);
		FEMTreeNode* node = &root;
		int d=0;
		while( d<maxDepth )
		{
			if( !node->children ) node->template initChildren< ThreadSafe >( nodeAllocator , NodeInitializer );
			int cIndex = FEMTreeNode::ChildIndex( center , p );
			node = node->children + cIndex;
			d++;
			width /= 2;
			for( int d=0 ; d<Dim ; d++ )
				if( (cIndex>>d) & 1 ) center[d] += width/2;
				else                  center[d] -= width/2;
		}
		return node;
	};

	size_t sCount = 0;

	for( size_t i=0 ; i<subSimplices.size() ; i++ )
	{
		// Find the finest depth at which the simplex is entirely within a node
		int tDepth;
		RegularGridIndex idx0 , idx;
		for( tDepth=0 ; tDepth<maxDepth ; tDepth++ )
		{
			// Get the grid index of the first vertex of the simplex
			for( int d=0 ; d<Dim ; d++ ) idx0.idx[d] = (int)( subSimplices[i][0][d] * (1<<(tDepth+1)) );
			bool done = false;
			for( int k=0 ; k<Dim && !done ; k++ )
			{
				for( int d=0 ; d<Dim ; d++ ) idx.idx[d] = (int)( subSimplices[i][k][d] * (1<<(tDepth+1)) );
				if( idx!=idx0 ) done = true;
			}
			if( done ) break;
		}

		// Add the simplex to the node
		FEMTreeNode* subSimplexNode = Leaf( subSimplices[i].center() , tDepth );
		for( size_t i=0 ; i<subSimplices.size() ; i++ ) sCount += _AddSimplex< ThreadSafe >( subSimplexNode , subSimplices[i] , maxDepth , simplices , nodeToIndexMap , nodeAllocator , NodeInitializer );
	}
	return sCount;
}
template< unsigned int Dim , class Real >
template< bool ThreadSafe >
size_t FEMTreeInitializer< Dim , Real >::_AddSimplex( FEMTreeNode* node , Simplex< Real , Dim , Dim-1 >& s , int maxDepth , std::vector< NodeSimplices< Dim , Real > >& simplices , std::vector< node_index_type >& nodeToIndexMap , Allocator< FEMTreeNode >* nodeAllocator , std::function< void ( FEMTreeNode& ) > NodeInitializer )
{
	int d = node->depth();
	if( d==maxDepth )
	{
		// If the simplex has non-zero size, add it to the list
		Real weight = s.measure();
		if( weight && weight==weight )
		{
			node_index_type nodeIndex = node->nodeData.nodeIndex;
			if( nodeIndex>=nodeToIndexMap.size() ) nodeToIndexMap.resize( nodeIndex+1 , -1 );
			node_index_type idx = nodeToIndexMap[ nodeIndex ];
			if( idx==-1 )
			{
				idx = (node_index_type)simplices.size();
				nodeToIndexMap[ nodeIndex ] = idx;
				simplices.resize( idx+1 );
				simplices[idx].node = node;
			}
			simplices[idx].data.push_back( s );
		}
		return 1;
	}
	else
	{
		size_t sCount = 0;
		if( !node->children ) node->template initChildren< ThreadSafe >( nodeAllocator , NodeInitializer );

		// Split up the simplex and pass the parts on to the children
		Point< Real , Dim > center;
		Real width;
		node->centerAndWidth( center , width );

		std::vector< std::vector< Simplex< Real , Dim , Dim-1 > > > childSimplices( 1 );
		childSimplices[0].push_back( s );
		for( int d=0 ; d<Dim ; d++ )
		{
			Point< Real , Dim > n ; n[Dim-d-1] = 1;
			std::vector< std::vector< Simplex< Real , Dim , Dim-1 > > > temp( (int)( 1<<(d+1) ) );
			for( int c=0 ; c<(1<<d) ; c++ ) for( int i=0 ; i<childSimplices[c].size() ; i++ ) childSimplices[c][i].split( n , center[Dim-d-1] , temp[2*c] , temp[2*c+1] );
			childSimplices = temp;
		}
		for( int c=0 ; c<(1<<Dim) ; c++ ) for( int i=0 ; i<childSimplices[c].size() ; i++ ) sCount += _AddSimplex< ThreadSafe >( node->children+c , childSimplices[c][i] , maxDepth , simplices , nodeToIndexMap , nodeAllocator , NodeInitializer );
		return sCount;
	}
}

template< unsigned int Dim , class Real >
template< class Data , class _Data , bool Dual >
size_t FEMTreeInitializer< Dim , Real >::Initialize( FEMTreeNode& root , ConstPointer( Data ) values , ConstPointer( int ) labels , int resolution[Dim] , std::vector< NodeSample< Dim , _Data > > derivatives[Dim] , Allocator< FEMTreeNode >* nodeAllocator , std::function< void ( FEMTreeNode& ) > NodeInitializer , std::function< _Data ( const Data& ) > DataConverter )
{
	auto Leaf = [&]( FEMTreeNode& root , const int idx[Dim] , int maxDepth )
	{
		for( int d=0 ; d<Dim ; d++ ) if( idx[d]<0 || idx[d]>=(1<<maxDepth) ) return (FEMTreeNode*)NULL;
		FEMTreeNode* node = &root;
		for( int d=0 ; d<maxDepth ; d++ )
		{
			if( !node->children ) node->template initChildren< false >( nodeAllocator , NodeInitializer );
			int cIndex = 0;
			for( int dd=0 ; dd<Dim ; dd++ ) if( idx[dd]&(1<<(maxDepth-d-1)) ) cIndex |= 1<<dd;
			node = node->children + cIndex;
		}
		return node;
	};
	auto FactorIndex = []( size_t i , const int resolution[Dim] , int idx[Dim] )
	{
		size_t ii = i;
		for( int d=0 ; d<Dim ; d++ ) idx[d] = ii % resolution[d] , ii /= resolution[d];
	};
	auto MakeIndex = [] ( const int idx[Dim] , const int resolution[Dim] )
	{
		size_t i = 0;
		for( int d=0 ; d<Dim ; d++ ) i = i * resolution[Dim-1-d] + idx[Dim-1-d];
		return i;
	};


	int maxResolution = resolution[0];
	for( int d=1 ; d<Dim ; d++ ) maxResolution = std::max< int >( maxResolution , resolution[d] );
	int maxDepth = 0;
	while( ( (1<<maxDepth) + ( Dual ? 0 : 1 ) )<maxResolution ) maxDepth++;

	size_t totalRes = 1;
	for( int d=0 ; d<Dim ; d++ ) totalRes *= resolution[d];

	// Iterate over each direction
	for( int d=0 ; d<Dim ; d++ ) for( size_t i=0 ; i<totalRes ; i++ )
	{
		// Factor the index into directional components and get the index of the next cell
		int idx[Dim] ; FactorIndex( i , resolution , idx ) ; idx[d]++;

		if( idx[d]<resolution[d] )
		{
			// Get the index of the next cell
			size_t ii = MakeIndex( idx , resolution );

			// [NOTE] There are no derivatives across negative labels
			if( labels[i]!=labels[ii] && labels[i]>=0 && labels[ii]>=0 )
			{
				if( !Dual ) idx[d]--;
				NodeSample< Dim , _Data > nodeSample;
				nodeSample.node = Leaf( root , idx , maxDepth );
				nodeSample.data = DataConverter( values[ii] ) - DataConverter( values[i] );
				if( nodeSample.node ) derivatives[d].push_back( nodeSample );
			}
		}
	}
	return maxDepth;
}

template< unsigned int Dim , class Real >
template< bool Dual , class Data >
unsigned int FEMTreeInitializer< Dim , Real >::Initialize( FEMTreeNode& root , DerivativeStream< Data >& dStream , std::vector< NodeSample< Dim , Data > > derivatives[Dim] , Allocator< FEMTreeNode >* nodeAllocator , std::function< void ( FEMTreeNode& ) > NodeInitializer )
{
	// Note:
	// --   Dual: The difference between [i] and [i+1] is stored at cell [i+1]
	// -- Primal: The difference between [i] and [i+1] is stored at cell [i]

	// Find the leaf containing the specified cell index
	auto Leaf = [&]( FEMTreeNode& root , const unsigned int idx[Dim] , unsigned int maxDepth )
	{
		for( int d=0 ; d<Dim ; d++ ) if( idx[d]<0 || idx[d]>=(unsigned int)(1<<maxDepth) ) return (FEMTreeNode*)NULL;
		FEMTreeNode* node = &root;
		for( unsigned int d=0 ; d<maxDepth ; d++ )
		{
			if( !node->children ) node->template initChildren< false >( nodeAllocator , NodeInitializer );
			int cIndex = 0;
			for( int dd=0 ; dd<Dim ; dd++ ) if( idx[dd]&(1<<(maxDepth-d-1)) ) cIndex |= 1<<dd;
			node = node->children + cIndex;
		}
		return node;
	};

	unsigned int resolution[Dim];
	dStream.resolution( resolution );
	unsigned int maxResolution = resolution[0];
	for( int d=1 ; d<Dim ; d++ ) maxResolution = std::max< unsigned int >( maxResolution , resolution[d] );
	unsigned int maxDepth = 0;

	// If we are using a dual formulation, we need at least maxResolution cells.
	// Otherwise, we need at least maxResolution-1 cells.
	while( (unsigned int)( (1<<maxDepth) + ( Dual ? 0 : 1 ) )<maxResolution ) maxDepth++;

	unsigned int idx[Dim] , dir;
	Data dValue;
	while( dStream.nextDerivative( idx , dir , dValue ) )
	{
		if( Dual ) idx[dir]++;
		NodeSample< Dim , Data > nodeSample;
		nodeSample.node = Leaf( root , idx , maxDepth );
		nodeSample.data = dValue;
		if( nodeSample.node ) derivatives[dir].push_back( nodeSample );
	}
	return maxDepth;
}
