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

#include <sstream>
#include <sstream>
#include <iomanip>
#include <unordered_map>
#include "MyMiscellany.h"
#include "MarchingCubes.h"
#include "MAT.h"


// Specialized iso-surface extraction
template< class Real , class Vertex >
struct IsoSurfaceExtractor< 3 , Real , Vertex >
{
	static const unsigned int Dim = 3;
	typedef typename FEMTree< Dim , Real >::LocalDepth LocalDepth;
	typedef typename FEMTree< Dim , Real >::LocalOffset LocalOffset;
	typedef typename FEMTree< Dim , Real >::ConstOneRingNeighborKey ConstOneRingNeighborKey;
	typedef typename FEMTree< Dim , Real >::ConstOneRingNeighbors ConstOneRingNeighbors;
	typedef RegularTreeNode< Dim , FEMTreeNodeData , depth_and_offset_type > TreeNode;
	template< unsigned int WeightDegree > using DensityEstimator = typename FEMTree< Dim , Real >::template DensityEstimator< WeightDegree >;
	template< typename FEMSigPack , unsigned int PointD > using _Evaluator = typename FEMTree< Dim , Real >::template _Evaluator< FEMSigPack , PointD >;
protected:
	static std::mutex _pointInsertionMutex;
	static std::atomic< size_t > _BadRootCount;
	//////////
	// _Key //
	//////////
	struct _Key
	{
		int idx[Dim];

		_Key( void ){ for( unsigned int d=0 ; d<Dim ; d++ ) idx[d] = 0; }

		int &operator[]( int i ){ return idx[i]; }
		const int &operator[]( int i ) const { return idx[i]; }

		bool operator == ( const _Key &key ) const
		{
			for( unsigned int d=0 ; d<Dim ; d++ ) if( idx[d]!=key[d] ) return false;
			return true;
		}
		bool operator != ( const _Key &key ) const { return !operator==( key ); }

		std::string to_string( void ) const
		{
			std::stringstream stream;
			stream << "(";
			for( unsigned int d=0 ; d<Dim ; d++ )
			{
				if( d ) stream << ",";
				stream << idx[d];
			}
			stream << ")";
			return stream.str();
		}

		struct Hasher
		{
			size_t operator()( const _Key &i ) const
			{
				size_t hash = 0;
				for( unsigned int d=0 ; d<Dim ; d++ ) hash ^= i.idx[d];
				return hash;
			}
		};
	};

	//////////////
	// _IsoEdge //
	//////////////
	struct _IsoEdge
	{
		_Key vertices[2];
		_IsoEdge( void ) {}
		_IsoEdge( _Key v1 , _Key v2 ){ vertices[0] = v1 , vertices[1] = v2; }
		_Key &operator[]( int idx ){ return vertices[idx]; }
		const _Key &operator[]( int idx ) const { return vertices[idx]; }
	};

	////////////////
	// _FaceEdges //
	////////////////
	struct _FaceEdges{ _IsoEdge edges[2] ; int count; };

	///////////////
	// SliceData //
	///////////////
	class SliceData
	{
		typedef RegularTreeNode< Dim , FEMTreeNodeData , depth_and_offset_type > TreeOctNode;
	public:
		template< unsigned int Indices >
		struct  _Indices
		{
			node_index_type idx[Indices];
			_Indices( void ){ for( unsigned int i=0 ; i<Indices ; i++ ) idx[i] = -1; }
			node_index_type& operator[] ( int i ) { return idx[i]; }
			const node_index_type& operator[] ( int i ) const { return idx[i]; }
		};
		typedef _Indices< HyperCube::Cube< Dim-1 >::template ElementNum< 0 >() > SquareCornerIndices;
		typedef _Indices< HyperCube::Cube< Dim-1 >::template ElementNum< 1 >() > SquareEdgeIndices;
		typedef _Indices< HyperCube::Cube< Dim-1 >::template ElementNum< 2 >() > SquareFaceIndices;

		struct SliceTableData
		{
			Pointer( SquareCornerIndices ) cTable;
			Pointer( SquareEdgeIndices   ) eTable;
			Pointer( SquareFaceIndices   ) fTable;
			node_index_type nodeOffset;
			node_index_type cCount , eCount , fCount;
			node_index_type nodeCount;
			SliceTableData( void ){ fCount = eCount = cCount = 0 , _oldNodeCount = 0 , cTable = NullPointer( SquareCornerIndices ) , eTable = NullPointer( SquareEdgeIndices ) , fTable = NullPointer( SquareFaceIndices ) , _cMap = _eMap = _fMap = NullPointer( node_index_type ) , _processed = NullPointer( char ); }
			void clear( void ){ DeletePointer( cTable ) ; DeletePointer( eTable ) ; DeletePointer( fTable ) ; DeletePointer( _cMap ) ; DeletePointer( _eMap ) ; DeletePointer( _fMap ) ; DeletePointer( _processed ) ; fCount = eCount = cCount = 0; }
			~SliceTableData( void ){ clear(); }

			SquareCornerIndices& cornerIndices( const TreeOctNode* node )             { return cTable[ node->nodeData.nodeIndex - nodeOffset ]; }
			SquareCornerIndices& cornerIndices( node_index_type idx )                 { return cTable[ idx - nodeOffset ]; }
			const SquareCornerIndices& cornerIndices( const TreeOctNode* node ) const { return cTable[ node->nodeData.nodeIndex - nodeOffset ]; }
			const SquareCornerIndices& cornerIndices( node_index_type idx )     const { return cTable[ idx - nodeOffset ]; }
			SquareEdgeIndices& edgeIndices( const TreeOctNode* node )                 { return eTable[ node->nodeData.nodeIndex - nodeOffset ]; }
			SquareEdgeIndices& edgeIndices( node_index_type idx )                     { return eTable[ idx - nodeOffset ]; }
			const SquareEdgeIndices& edgeIndices( const TreeOctNode* node )     const { return eTable[ node->nodeData.nodeIndex - nodeOffset ]; }
			const SquareEdgeIndices& edgeIndices( node_index_type idx )         const { return eTable[ idx - nodeOffset ]; }
			SquareFaceIndices& faceIndices( const TreeOctNode* node )                 { return fTable[ node->nodeData.nodeIndex - nodeOffset ]; }
			SquareFaceIndices& faceIndices( node_index_type idx )                     { return fTable[ idx - nodeOffset ]; }
			const SquareFaceIndices& faceIndices( const TreeOctNode* node )     const { return fTable[ node->nodeData.nodeIndex - nodeOffset ]; }
			const SquareFaceIndices& faceIndices( node_index_type idx )         const { return fTable[ idx - nodeOffset ]; }

		protected:
			Pointer( node_index_type ) _cMap;
			Pointer( node_index_type ) _eMap;
			Pointer( node_index_type ) _fMap;
			Pointer( char ) _processed;
			node_index_type _oldNodeCount;
			friend SliceData;
		};
		struct XSliceTableData
		{
			Pointer( SquareCornerIndices ) eTable;
			Pointer( SquareEdgeIndices ) fTable;
			node_index_type nodeOffset;
			node_index_type fCount , eCount;
			node_index_type nodeCount;
			XSliceTableData( void ){ fCount = eCount = 0 , _oldNodeCount = 0 , eTable = NullPointer( SquareCornerIndices ) , fTable = NullPointer( SquareEdgeIndices ) , _eMap = _fMap = NullPointer( node_index_type ); }
			~XSliceTableData( void ){ clear(); }
			void clear( void ) { DeletePointer( fTable ) ; DeletePointer( eTable ) ; DeletePointer( _eMap ) ; DeletePointer( _fMap ) ; fCount = eCount = 0; }

			SquareCornerIndices& edgeIndices( const TreeOctNode* node )             { return eTable[ node->nodeData.nodeIndex - nodeOffset ]; }
			SquareCornerIndices& edgeIndices( node_index_type idx )                 { return eTable[ idx - nodeOffset ]; }
			const SquareCornerIndices& edgeIndices( const TreeOctNode* node ) const { return eTable[ node->nodeData.nodeIndex - nodeOffset ]; }
			const SquareCornerIndices& edgeIndices( node_index_type idx )     const { return eTable[ idx - nodeOffset ]; }
			SquareEdgeIndices& faceIndices( const TreeOctNode* node )               { return fTable[ node->nodeData.nodeIndex - nodeOffset ]; }
			SquareEdgeIndices& faceIndices( node_index_type idx )                   { return fTable[ idx - nodeOffset ]; }
			const SquareEdgeIndices& faceIndices( const TreeOctNode* node )   const { return fTable[ node->nodeData.nodeIndex - nodeOffset ]; }
			const SquareEdgeIndices& faceIndices( node_index_type idx )       const { return fTable[ idx - nodeOffset ]; }
		protected:
			Pointer( node_index_type ) _eMap;
			Pointer( node_index_type ) _fMap;
			node_index_type _oldNodeCount;
			friend SliceData;
		};
		template< unsigned int D , unsigned int ... Ks > struct HyperCubeTables{};
		template< unsigned int D , unsigned int K >
		struct HyperCubeTables< D , K >
		{
			static unsigned int CellOffset[ HyperCube::Cube< D >::template ElementNum< K >() ][ HyperCube::Cube< D >::template IncidentCubeNum< K >() ];
			static unsigned int IncidentElementCoIndex[ HyperCube::Cube< D >::template ElementNum< K >() ][ HyperCube::Cube< D >::template IncidentCubeNum< K >() ];
			static unsigned int CellOffsetAntipodal[ HyperCube::Cube< D >::template ElementNum< K >() ];
			static typename HyperCube::Cube< D >::template IncidentCubeIndex< K > IncidentCube[ HyperCube::Cube< D >::template ElementNum< K >() ];
			static typename HyperCube::Direction Directions[ HyperCube::Cube< D >::template ElementNum< K >() ][ D ];
			static void SetTables( void )
			{
				for( typename HyperCube::Cube< D >::template Element< K > e ; e<HyperCube::Cube< D >::template ElementNum< K >() ; e++ )
				{
					for( typename HyperCube::Cube< D >::template IncidentCubeIndex< K > i ; i<HyperCube::Cube< D >::template IncidentCubeNum< K >() ; i++ )
					{
						CellOffset[e.index][i.index] = HyperCube::Cube< D >::CellOffset( e , i );
						IncidentElementCoIndex[e.index][i.index] = HyperCube::Cube< D >::IncidentElement( e , i ).coIndex();
					}
					CellOffsetAntipodal[e.index] = HyperCube::Cube< D >::CellOffset( e , HyperCube::Cube< D >::IncidentCube( e ).antipodal() );
					IncidentCube[ e.index ] = HyperCube::Cube< D >::IncidentCube( e );
					e.directions( Directions[e.index] );
				}
			}
		};
		template< unsigned int D , unsigned int K1 , unsigned int K2 >
		struct HyperCubeTables< D , K1 , K2 >
		{
			static typename HyperCube::Cube< D >::template Element< K2 > OverlapElements[ HyperCube::Cube< D >::template ElementNum< K1 >() ][ HyperCube::Cube< D >::template OverlapElementNum< K1 , K2 >() ];
			static bool Overlap[ HyperCube::Cube< D >::template ElementNum< K1 >() ][ HyperCube::Cube< D >::template ElementNum< K2 >() ];
			static void SetTables( void )
			{
				for( typename HyperCube::Cube< D >::template Element< K1 > e ; e<HyperCube::Cube< D >::template ElementNum< K1 >() ; e++ )
				{
					for( typename HyperCube::Cube< D >::template Element< K2 > _e ; _e<HyperCube::Cube< D >::template ElementNum< K2 >() ; _e++ )
						Overlap[e.index][_e.index] = HyperCube::Cube< D >::Overlap( e , _e );
					HyperCube::Cube< D >::OverlapElements( e , OverlapElements[e.index] );
				}
				if( !K2 ) HyperCubeTables< D , K1 >::SetTables();
			}
		};

		template< unsigned int D=Dim , unsigned int K1=Dim , unsigned int K2=Dim > static typename std::enable_if<                 K2!=0 >::type SetHyperCubeTables( void )
		{
			HyperCubeTables< D , K1 , K2 >::SetTables() ; SetHyperCubeTables< D , K1 , K2-1 >();
		}
		template< unsigned int D=Dim , unsigned int K1=Dim , unsigned int K2=Dim > static typename std::enable_if<        K1!=0 && K2==0 >::type SetHyperCubeTables( void )
		{
			HyperCubeTables< D , K1 , K2 >::SetTables(); SetHyperCubeTables< D , K1-1 , D >();
		}
		template< unsigned int D=Dim , unsigned int K1=Dim , unsigned int K2=Dim > static typename std::enable_if< D!=1 && K1==0 && K2==0 >::type SetHyperCubeTables( void )
		{
			HyperCubeTables< D , K1 , K2 >::SetTables() ; SetHyperCubeTables< D-1 , D-1 , D-1 >();
		}
		template< unsigned int D=Dim , unsigned int K1=Dim , unsigned int K2=Dim > static typename std::enable_if< D==1 && K1==0 && K2==0 >::type SetHyperCubeTables( void )
		{
			HyperCubeTables< D , K1 , K2 >::SetTables();
		}

		static void SetSliceTableData( const SortedTreeNodes< Dim >& sNodes , SliceTableData* sData0 , XSliceTableData* xData , SliceTableData* sData1 , int depth , int offset )
		{
			// [NOTE] This is structure is purely for determining adjacency and is independent of the FEM degree
			typedef typename FEMTree< Dim , Real >::ConstOneRingNeighborKey ConstOneRingNeighborKey;
			if( offset<0 || offset>((size_t)1<<depth) ) return;
			if( sData0 )
			{
				std::pair< node_index_type , node_index_type > span( sNodes.begin( depth , offset-1 ) , sNodes.end( depth , offset ) );
				sData0->nodeOffset = span.first , sData0->nodeCount = span.second - span.first;
			}
			if( sData1 )
			{
				std::pair< node_index_type , node_index_type > span( sNodes.begin( depth , offset ) , sNodes.end( depth , offset+1 ) );
				sData1->nodeOffset = span.first , sData1->nodeCount = span.second - span.first;
			}
			if( xData )
			{
				std::pair< node_index_type , node_index_type > span( sNodes.begin( depth , offset ) , sNodes.end( depth , offset ) );
				xData->nodeOffset = span.first , xData->nodeCount = span.second - span.first;
			}
			SliceTableData* sData[] = { sData0 , sData1 };
			for( int i=0 ; i<2 ; i++ ) if( sData[i] )
			{
				if( sData[i]->nodeCount>sData[i]->_oldNodeCount )
				{
					DeletePointer( sData[i]->_cMap ) ; DeletePointer( sData[i]->_eMap ) ; DeletePointer( sData[i]->_fMap );
					DeletePointer( sData[i]->cTable ) ; DeletePointer( sData[i]->eTable ) ; DeletePointer( sData[i]->fTable );
					DeletePointer( sData[i]->_processed );
					sData[i]->_cMap = NewPointer< node_index_type >( sData[i]->nodeCount * HyperCube::Cube< Dim-1 >::template ElementNum< 0 >() );
					sData[i]->_eMap = NewPointer< node_index_type >( sData[i]->nodeCount * HyperCube::Cube< Dim-1 >::template ElementNum< 1 >() );
					sData[i]->_fMap = NewPointer< node_index_type >( sData[i]->nodeCount * HyperCube::Cube< Dim-1 >::template ElementNum< 2 >() );
					sData[i]->_processed = NewPointer< char >( sData[i]->nodeCount );
					sData[i]->cTable = NewPointer< typename SliceData::SquareCornerIndices >( sData[i]->nodeCount );
					sData[i]->eTable = NewPointer< typename SliceData::SquareEdgeIndices >( sData[i]->nodeCount );
					sData[i]->fTable = NewPointer< typename SliceData::SquareFaceIndices >( sData[i]->nodeCount );
					sData[i]->_oldNodeCount = sData[i]->nodeCount;
				}
				memset( sData[i]->_cMap , 0 , sizeof(node_index_type) * sData[i]->nodeCount * HyperCube::Cube< Dim-1 >::template ElementNum< 0 >() );
				memset( sData[i]->_eMap , 0 , sizeof(node_index_type) * sData[i]->nodeCount * HyperCube::Cube< Dim-1 >::template ElementNum< 1 >() );
				memset( sData[i]->_fMap , 0 , sizeof(node_index_type) * sData[i]->nodeCount * HyperCube::Cube< Dim-1 >::template ElementNum< 2 >() );
				memset( sData[i]->_processed , 0 , sizeof(char) * sData[i]->nodeCount );
			}
			if( xData )
			{
				if( xData->nodeCount>xData->_oldNodeCount )
				{
					DeletePointer( xData->_eMap ) ; DeletePointer( xData->_fMap );
					DeletePointer( xData->eTable ) ; DeletePointer( xData->fTable );
					xData->_eMap = NewPointer< node_index_type >( xData->nodeCount * HyperCube::Cube< Dim-1 >::template ElementNum< 0 >() );
					xData->_fMap = NewPointer< node_index_type >( xData->nodeCount * HyperCube::Cube< Dim-1 >::template ElementNum< 1 >() );
					xData->eTable = NewPointer< typename SliceData::SquareCornerIndices >( xData->nodeCount );
					xData->fTable = NewPointer< typename SliceData::SquareEdgeIndices >( xData->nodeCount );
					xData->_oldNodeCount = xData->nodeCount;
				}
				memset( xData->_eMap , 0 , sizeof(node_index_type) * xData->nodeCount * HyperCube::Cube< Dim-1 >::template ElementNum< 0 >() );
				memset( xData->_fMap , 0 , sizeof(node_index_type) * xData->nodeCount * HyperCube::Cube< Dim-1 >::template ElementNum< 1 >() );
			}
			std::vector< ConstOneRingNeighborKey > neighborKeys( ThreadPool::NumThreads() );
			for( size_t i=0 ; i<neighborKeys.size() ; i++ ) neighborKeys[i].set( depth );

			typedef typename FEMTree< Dim , Real >::ConstOneRingNeighbors ConstNeighbors;

			// Process the corners
			// z: which side of the cell	\in {0,1}
			// zOff: which neighbor			\in {-1,0,1}
			auto ProcessCorners = []( SliceTableData& sData , const ConstNeighbors& neighbors , HyperCube::Direction zDir , int zOff )
			{
				const TreeOctNode* node = neighbors.neighbors[1][1][1+zOff];
				node_index_type i = node->nodeData.nodeIndex;
				// Iterate over the corners in the face
				for( typename HyperCube::Cube< Dim-1 >::template Element< 0 > _c ; _c<HyperCube::Cube< Dim-1 >::template ElementNum< 0 >() ; _c++ )
				{
					bool owner = true;

					typename HyperCube::Cube< Dim >::template Element< 0 > c( zDir , _c.index );																	// Corner-in-cube index
					typename HyperCube::Cube< Dim >::template IncidentCubeIndex< 0 > my_ic = HyperCubeTables< Dim , 0 >::IncidentCube[c.index];						// The index of the node relative to the corner
					for( typename HyperCube::Cube< Dim >::template IncidentCubeIndex< 0 > ic ; ic<HyperCube::Cube< Dim >::template IncidentCubeNum< 0 >() ; ic++ )	// Iterate over the nodes adjacent to the corner
					{
						// Get the index of cube relative to the corner neighbors
						unsigned int xx = HyperCubeTables< Dim , 0 >::CellOffset[c.index][ic.index] + zOff;
						// If the neighbor exists and comes before, they own the corner
						if( neighbors.neighbors.data[xx] && ic<my_ic ){ owner = false ; break; }
					}
					if( owner )
					{
						node_index_type myCount = ( i-sData.nodeOffset ) * HyperCube::Cube< Dim-1 >::template ElementNum< 0 >() + _c.index;
						sData._cMap[ myCount ] = 1;
						// Set the corner pointer for all cubes incident on the corner
						for( typename HyperCube::Cube< Dim >::template IncidentCubeIndex< 0 > ic ; ic<HyperCube::Cube< Dim >::template IncidentCubeNum< 0 >() ; ic++ )	// Iterate over the nodes adjacent to the corner
						{
							unsigned int xx = HyperCubeTables< Dim , 0 >::CellOffset[c.index][ic.index] + zOff;
							// If the neighbor exits, sets its corner
							if( neighbors.neighbors.data[xx] ) sData.cornerIndices( neighbors.neighbors.data[xx] )[ HyperCubeTables< Dim , 0 >::IncidentElementCoIndex[c.index][ic.index] ] = myCount;
						}
					}
				}
			};
			// Process the in-plane edges
			auto ProcessIEdges = []( SliceTableData& sData , const ConstNeighbors& neighbors , HyperCube::Direction zDir , int zOff )
			{
				const TreeOctNode* node = neighbors.neighbors[1][1][1+zOff];
				node_index_type i = node->nodeData.nodeIndex;
				// Iterate over the edges in the face
				for( typename HyperCube::Cube< Dim-1 >::template Element< 1 > _e ; _e<HyperCube::Cube< Dim-1 >::template ElementNum< 1 >() ; _e++ )
				{
					bool owner = true;

					// The edge in the cube
					typename HyperCube::Cube< Dim >::template Element< 1 > e( zDir , _e.index );
					// The index of the cube relative to the edge
					typename HyperCube::Cube< Dim >::template IncidentCubeIndex< 1 > my_ic = HyperCubeTables< Dim , 1 >::IncidentCube[e.index];
					// Iterate over the cubes incident on the edge
					for( typename HyperCube::Cube< Dim >::template IncidentCubeIndex< 1 > ic ; ic<HyperCube::Cube< Dim >::template IncidentCubeNum< 1 >() ; ic++ )
					{
						// Get the indices of the cube relative to the center
						unsigned int xx = HyperCubeTables< Dim , 1 >::CellOffset[e.index][ic.index] + zOff;
						// If the neighbor exists and comes before, they own the corner
						if( neighbors.neighbors.data[xx] && ic<my_ic ){ owner = false ; break; }
					}
					if( owner )
					{
						node_index_type myCount = ( i - sData.nodeOffset ) * HyperCube::Cube< Dim-1 >::template ElementNum< 1 >() + _e.index;
						sData._eMap[ myCount ] = 1;
						// Set all edge indices
						for( typename HyperCube::Cube< Dim >::template IncidentCubeIndex< 1 > ic ; ic<HyperCube::Cube< Dim >::template IncidentCubeNum< 1 >() ; ic++ )
						{
							unsigned int xx = HyperCubeTables< Dim , 1 >::CellOffset[e.index][ic.index] + zOff;
							// If the neighbor exists, set the index
							if( neighbors.neighbors.data[xx] ) sData.edgeIndices( neighbors.neighbors.data[xx] )[ HyperCubeTables< Dim , 1 >::IncidentElementCoIndex[e.index][ic.index] ] = myCount;
						}
					}
				}
			};
			// Process the cross-plane edges
			auto ProcessXEdges = []( XSliceTableData& xData , const ConstNeighbors& neighbors )
			{
				const TreeOctNode* node = neighbors.neighbors[1][1][1];
				node_index_type i = node->nodeData.nodeIndex;
				for( typename HyperCube::Cube< Dim-1 >::template Element< 0 > _c ; _c<HyperCube::Cube< Dim-1 >::template ElementNum< 0 >() ; _c++ )
				{
					bool owner = true;

					typename HyperCube::Cube< Dim >::template Element< 1 > e( HyperCube::CROSS , _c.index );
					typename HyperCube::Cube< Dim >::template IncidentCubeIndex< 1 > my_ic = HyperCubeTables< Dim , 1 >::IncidentCube[e.index];

					for( typename HyperCube::Cube< Dim >::template IncidentCubeIndex< 1 > ic ; ic<HyperCube::Cube< Dim >::template IncidentCubeNum< 1 >() ; ic++ )
					{
						unsigned int xx = HyperCubeTables< Dim , 1 >::CellOffset[e.index][ic.index];
						if( neighbors.neighbors.data[xx] && ic<my_ic ){ owner = false ; break; }
					}
					if( owner )
					{
						node_index_type myCount = ( i - xData.nodeOffset ) * HyperCube::Cube< Dim-1 >::template ElementNum< 0 >() + _c.index;
						xData._eMap[ myCount ] = 1;

						// Set all edge indices
						for( typename HyperCube::Cube< Dim >::template IncidentCubeIndex< 1 > ic ; ic<HyperCube::Cube< Dim >::template IncidentCubeNum< 1 >() ; ic++ )
						{
							unsigned int xx = HyperCubeTables< Dim , 1 >::CellOffset[e.index][ic.index];
							if( neighbors.neighbors.data[xx] ) xData.edgeIndices( neighbors.neighbors.data[xx] )[ HyperCubeTables< Dim , 1 >::IncidentElementCoIndex[e.index][ic.index] ] = myCount;
						}
					}
				}
			};
			// Process the in-plane faces
			auto ProcessIFaces = []( SliceTableData& sData , const ConstNeighbors& neighbors , HyperCube::Direction zDir , int zOff )
			{
				const TreeOctNode* node = neighbors.neighbors[1][1][1+zOff];
				node_index_type i = node->nodeData.nodeIndex;
				for( typename HyperCube::Cube< Dim-1 >::template Element< 2 > _f ; _f<HyperCube::Cube< Dim-1 >::template ElementNum< 2 >() ; _f++ )
				{
					bool owner = true;

					typename HyperCube::Cube< Dim >::template Element< 2 > f( zDir , _f.index );				
					typename HyperCube::Cube< Dim >::template IncidentCubeIndex< 2 > my_ic = HyperCubeTables< Dim , 2 >::IncidentCube[f.index];

					for( typename HyperCube::Cube< Dim >::template IncidentCubeIndex< 2 > ic ; ic<HyperCube::Cube< Dim >::template IncidentCubeNum< 2 >() ; ic++ )
					{
						unsigned int xx = HyperCubeTables< Dim , 2 >::CellOffset[f.index][ic.index] + zOff;
						if( neighbors.neighbors.data[xx] && ic<my_ic ){ owner = false ; break; }
					}
					if( owner )
					{
						node_index_type myCount = ( i - sData.nodeOffset ) * HyperCube::Cube< Dim-1 >::template ElementNum< 2 >() + _f.index;
						sData._fMap[ myCount ] = 1;

						// Set all the face indices
						for( typename HyperCube::Cube< Dim >::template IncidentCubeIndex< 2 > ic ; ic<HyperCube::Cube< Dim >::template IncidentCubeNum< 2 >() ; ic++ )
						{
							unsigned int xx = HyperCubeTables< Dim , 2 >::CellOffset[f.index][ic.index] + zOff;
							if( neighbors.neighbors.data[xx] ) sData.faceIndices( neighbors.neighbors.data[xx] )[ HyperCubeTables< Dim , 2 >::IncidentElementCoIndex[f.index][ic.index] ] = myCount;
						}
					}
				}
			};

			// Process the cross-plane faces
			auto ProcessXFaces = []( XSliceTableData& xData , const ConstNeighbors& neighbors )
			{
				const TreeOctNode* node = neighbors.neighbors[1][1][1];
				node_index_type i = node->nodeData.nodeIndex;
				for( typename HyperCube::Cube< Dim-1 >::template Element< 1 > _e ; _e<HyperCube::Cube< Dim-1 >::template ElementNum< 1 >() ; _e++ )
				{
					bool owner = true;

					typename HyperCube::Cube< Dim >::template Element< 2 > f( HyperCube::CROSS , _e.index );				
					typename HyperCube::Cube< Dim >::template IncidentCubeIndex< 2 > my_ic = HyperCubeTables< Dim , 2 >::IncidentCube[f.index];

					for( typename HyperCube::Cube< Dim >::template IncidentCubeIndex< 2 > ic ; ic<HyperCube::Cube< Dim >::template IncidentCubeNum< 2 >() ; ic++ )
					{
						unsigned int xx = HyperCubeTables< Dim , 2 >::CellOffset[f.index][ic.index];
						if( neighbors.neighbors.data[xx] && ic<my_ic ){ owner = false ; break; }
					}
					if( owner )
					{
						node_index_type myCount = ( i - xData.nodeOffset ) * HyperCube::Cube< Dim-1 >::template ElementNum< 1 >() + _e.index;
						xData._fMap[ myCount ] = 1;

						// Set all the face indices
						for( typename HyperCube::Cube< Dim >::template IncidentCubeIndex< 2 > ic ; ic<HyperCube::Cube< Dim >::template IncidentCubeNum< 2 >() ; ic++ )
						{
							unsigned int xx = HyperCubeTables< Dim , 2 >::CellOffset[f.index][ic.index];
							if( neighbors.neighbors.data[xx] ) xData.faceIndices( neighbors.neighbors.data[xx] )[ HyperCubeTables< Dim , 2 >::IncidentElementCoIndex[f.index][ic.index] ] = myCount;
						}
					}
				}
			};

			// Try and get at the nodes outside of the slab through the neighbor key
			ThreadPool::Parallel_for( sNodes.begin(depth,offset) , sNodes.end(depth,offset) , [&]( unsigned int thread , size_t i )
			{
				ConstOneRingNeighborKey& neighborKey = neighborKeys[ thread ];
				const TreeOctNode* node = sNodes.treeNodes[i];
				ConstNeighbors& neighbors = neighborKey.getNeighbors( node );
				for( int i=0 ; i<3 ; i++ ) for( int j=0 ; j<3 ; j++ ) for( int k=0 ; k<3 ; k++ ) if( !IsActiveNode< Dim >( neighbors.neighbors[i][j][k] ) ) neighbors.neighbors[i][j][k] = NULL;

				if( sData0 )
				{
					ProcessCorners( *sData0 , neighbors , HyperCube::BACK , 0 ) , ProcessIEdges( *sData0 , neighbors , HyperCube::BACK , 0 ) , ProcessIFaces( *sData0 , neighbors , HyperCube::BACK , 0 );
					const TreeOctNode* _node = neighbors.neighbors[1][1][0];
					if( _node )
					{
						ProcessCorners( *sData0 , neighbors , HyperCube::FRONT , -1 ) , ProcessIEdges( *sData0 , neighbors , HyperCube::FRONT , -1 ) , ProcessIFaces( *sData0 , neighbors , HyperCube::FRONT , -1 );
						sData0->_processed[ _node->nodeData.nodeIndex - sNodes.begin(depth,offset-1) ] = 1;
					}
				}
				if( sData1 )
				{
					ProcessCorners( *sData1 , neighbors , HyperCube::FRONT , 0 ) , ProcessIEdges( *sData1 , neighbors , HyperCube::FRONT , 0 ) , ProcessIFaces( *sData1 , neighbors , HyperCube::FRONT , 0 );
					const TreeOctNode* _node = neighbors.neighbors[1][1][2];
					if( _node )
					{
						ProcessCorners( *sData1 , neighbors , HyperCube::BACK , 1 ) , ProcessIEdges( *sData1 , neighbors , HyperCube::BACK , 1 ) , ProcessIFaces( *sData1, neighbors , HyperCube::BACK , 1 );
						sData1->_processed[ _node->nodeData.nodeIndex - sNodes.begin(depth,offset+1) ] = true;
					}
				}
				if( xData ) ProcessXEdges( *xData , neighbors ) , ProcessXFaces( *xData , neighbors );
			}
			);
			if( sData0 )
			{
				node_index_type off = sNodes.begin(depth,offset-1);
				node_index_type size = sNodes.end(depth,offset-1) - sNodes.begin(depth,offset-1);
				ThreadPool::Parallel_for( 0 , size , [&]( unsigned int thread , size_t i )
				{
					if( !sData0->_processed[i] )
					{
						ConstOneRingNeighborKey& neighborKey = neighborKeys[ thread ];
						const TreeOctNode* node = sNodes.treeNodes[i+off];
						ConstNeighbors& neighbors = neighborKey.getNeighbors( node );
						for( int i=0 ; i<3 ; i++ ) for( int j=0 ; j<3 ; j++ ) for( int k=0 ; k<3 ; k++ ) if( !IsActiveNode< Dim >( neighbors.neighbors[i][j][k] ) ) neighbors.neighbors[i][j][k] = NULL;
						ProcessCorners( *sData0 , neighbors , HyperCube::FRONT , 0 ) , ProcessIEdges( *sData0 , neighbors , HyperCube::FRONT , 0 ) , ProcessIFaces( *sData0 , neighbors , HyperCube::FRONT , 0 );
					}
				}
				);
			}
			if( sData1 )
			{
				node_index_type off = sNodes.begin(depth,offset+1);
				node_index_type size = sNodes.end(depth,offset+1) - sNodes.begin(depth,offset+1);
				ThreadPool::Parallel_for( 0 , size , [&]( unsigned int thread , size_t i )
				{
					if( !sData1->_processed[i] )
					{
						ConstOneRingNeighborKey& neighborKey = neighborKeys[ thread ];
						const TreeOctNode* node = sNodes.treeNodes[i+off];
						ConstNeighbors& neighbors = neighborKey.getNeighbors( node );
						for( int i=0 ; i<3 ; i++ ) for( int j=0 ; j<3 ; j++ ) for( int k=0 ; k<3 ; k++ ) if( !IsActiveNode< Dim >( neighbors.neighbors[i][j][k] ) ) neighbors.neighbors[i][j][k] = NULL;
						ProcessCorners( *sData1 , neighbors , HyperCube::BACK , 0 ) , ProcessIEdges( *sData1 , neighbors , HyperCube::BACK , 0 ) , ProcessIFaces( *sData1 , neighbors , HyperCube::BACK , 0 );
					}
				}
				);
			}

			auto SetICounts = [&]( SliceTableData& sData )
			{
				node_index_type cCount = 0 , eCount = 0 , fCount = 0;

				for( node_index_type i=0 ; i<sData.nodeCount * (node_index_type)HyperCube::Cube< Dim-1 >::template ElementNum< 0 >() ; i++ ) if( sData._cMap[i] ) sData._cMap[i] = cCount++;
				for( node_index_type i=0 ; i<sData.nodeCount * (node_index_type)HyperCube::Cube< Dim-1 >::template ElementNum< 1 >() ; i++ ) if( sData._eMap[i] ) sData._eMap[i] = eCount++;
				for( node_index_type i=0 ; i<sData.nodeCount * (node_index_type)HyperCube::Cube< Dim-1 >::template ElementNum< 2 >() ; i++ ) if( sData._fMap[i] ) sData._fMap[i] = fCount++;
				ThreadPool::Parallel_for( 0 , sData.nodeCount , [&]( unsigned int  , size_t i )
				{
					for( unsigned int j=0 ; j<HyperCube::Cube< Dim-1 >::template ElementNum< 0 >() ; j++ ) sData.cTable[i][j] = sData._cMap[ sData.cTable[i][j] ];
					for( unsigned int j=0 ; j<HyperCube::Cube< Dim-1 >::template ElementNum< 1 >() ; j++ ) sData.eTable[i][j] = sData._eMap[ sData.eTable[i][j] ];
					for( unsigned int j=0 ; j<HyperCube::Cube< Dim-1 >::template ElementNum< 2 >() ; j++ ) sData.fTable[i][j] = sData._fMap[ sData.fTable[i][j] ];
				}
				);
				sData.cCount = cCount , sData.eCount = eCount , sData.fCount = fCount;
			};
			auto SetXCounts = [&]( XSliceTableData& xData )
			{
				node_index_type eCount = 0 , fCount = 0;

				for( node_index_type i=0 ; i<xData.nodeCount * (node_index_type)HyperCube::Cube< Dim-1 >::template ElementNum< 0 >() ; i++ ) if( xData._eMap[i] ) xData._eMap[i] = eCount++;
				for( node_index_type i=0 ; i<xData.nodeCount * (node_index_type)HyperCube::Cube< Dim-1 >::template ElementNum< 1 >() ; i++ ) if( xData._fMap[i] ) xData._fMap[i] = fCount++;
				ThreadPool::Parallel_for( 0 , xData.nodeCount , [&]( unsigned int , size_t i )
				{
					for( unsigned int j=0 ; j<HyperCube::Cube< Dim-1 >::template ElementNum< 0 >() ; j++ ) xData.eTable[i][j] = xData._eMap[ xData.eTable[i][j] ];
					for( unsigned int j=0 ; j<HyperCube::Cube< Dim-1 >::template ElementNum< 1 >() ; j++ ) xData.fTable[i][j] = xData._fMap[ xData.fTable[i][j] ];
				}
				);
				xData.eCount = eCount , xData.fCount = fCount;
			};

			if( sData0 ) SetICounts( *sData0 );
			if( sData1 ) SetICounts( *sData1 );
			if( xData  ) SetXCounts( *xData  );
		}
	};

	//////////////////
	// _SliceValues //
	//////////////////
	struct _SliceValues
	{
		typename SliceData::SliceTableData sliceData;
		Pointer( Real ) cornerValues ; Pointer( Point< Real , Dim > ) cornerGradients ; Pointer( char ) cornerSet;
		Pointer( _Key ) edgeKeys ; Pointer( char ) edgeSet;
		Pointer( _FaceEdges ) faceEdges ; Pointer( char ) faceSet;
		Pointer( char ) mcIndices;
		std::unordered_map< _Key , std::vector< _IsoEdge > , typename _Key::Hasher > faceEdgeMap;
		std::unordered_map< _Key , std::pair< node_index_type, Vertex > , typename _Key::Hasher > edgeVertexMap;
		std::unordered_map< _Key , _Key , typename _Key::Hasher > vertexPairMap;
		std::vector< std::vector< std::pair< _Key , std::vector< _IsoEdge > > > > faceEdgeKeyValues;
		std::vector< std::vector< std::pair< _Key , std::pair< node_index_type , Vertex > > > > edgeVertexKeyValues;
		std::vector< std::vector< std::pair< _Key , _Key > > > vertexPairKeyValues;

		_SliceValues( void )
		{
			_oldCCount = _oldECount = _oldFCount = 0;
			_oldNCount = 0;
			cornerValues = NullPointer( Real ) ; cornerGradients = NullPointer( Point< Real , Dim > ) ; cornerSet = NullPointer( char );
			edgeKeys = NullPointer( _Key ) ; edgeSet = NullPointer( char );
			faceEdges = NullPointer( _FaceEdges ) ; faceSet = NullPointer( char );
			mcIndices = NullPointer( char );
			edgeVertexKeyValues.resize( ThreadPool::NumThreads() );
			vertexPairKeyValues.resize( ThreadPool::NumThreads() );
			faceEdgeKeyValues.resize( ThreadPool::NumThreads() );
		}
		~_SliceValues( void )
		{
			_oldCCount = _oldECount = _oldFCount = 0;
			_oldNCount = 0;
			FreePointer( cornerValues ) ; FreePointer( cornerGradients ) ; FreePointer( cornerSet );
			FreePointer( edgeKeys ) ; FreePointer( edgeSet );
			FreePointer( faceEdges ) ; FreePointer( faceSet );
			FreePointer( mcIndices );
		}
		void setEdgeVertexMap( void )
		{
			for( node_index_type i=0 ; i<(node_index_type)edgeVertexKeyValues.size() ; i++ )
			{
				for( int j=0 ; j<edgeVertexKeyValues[i].size() ; j++ ) edgeVertexMap[ edgeVertexKeyValues[i][j].first ] = edgeVertexKeyValues[i][j].second;
				edgeVertexKeyValues[i].clear();
			}
		}
		void setVertexPairMap( void )
		{
			for( node_index_type i=0 ; i<(node_index_type)vertexPairKeyValues.size() ; i++ )
			{
				for( int j=0 ; j<vertexPairKeyValues[i].size() ; j++ )
				{
					vertexPairMap[ vertexPairKeyValues[i][j].first ] = vertexPairKeyValues[i][j].second;
					vertexPairMap[ vertexPairKeyValues[i][j].second ] = vertexPairKeyValues[i][j].first;
				}
				vertexPairKeyValues[i].clear();
			}
		}
		void setFaceEdgeMap( void )
		{
			for( node_index_type i=0 ; i<(node_index_type)faceEdgeKeyValues.size() ; i++ )
			{
				for( int j=0 ; j<faceEdgeKeyValues[i].size() ; j++ )
				{
					auto iter = faceEdgeMap.find( faceEdgeKeyValues[i][j].first );
					if( iter==faceEdgeMap.end() ) faceEdgeMap[ faceEdgeKeyValues[i][j].first ] = faceEdgeKeyValues[i][j].second;
					else for( int k=0 ; k<faceEdgeKeyValues[i][j].second.size() ; k++ ) iter->second.push_back( faceEdgeKeyValues[i][j].second[k] );
				}
				faceEdgeKeyValues[i].clear();
			}
		}
		void reset( bool nonLinearFit )
		{
			faceEdgeMap.clear() , edgeVertexMap.clear() , vertexPairMap.clear();
			for( node_index_type i=0 ; i<(node_index_type)edgeVertexKeyValues.size() ; i++ ) edgeVertexKeyValues[i].clear();
			for( node_index_type i=0 ; i<(node_index_type)vertexPairKeyValues.size() ; i++ ) vertexPairKeyValues[i].clear();
			for( node_index_type i=0 ; i<(node_index_type)faceEdgeKeyValues.size() ; i++ ) faceEdgeKeyValues[i].clear();

			if( _oldNCount<sliceData.nodeCount )
			{
				_oldNCount = sliceData.nodeCount;
				FreePointer( mcIndices );
				if( sliceData.nodeCount>0 ) mcIndices = AllocPointer< char >( _oldNCount );
			}
			if( _oldCCount<sliceData.cCount )
			{
				_oldCCount = sliceData.cCount;
				FreePointer( cornerValues ) ; FreePointer( cornerGradients ) ; FreePointer( cornerSet );
				if( sliceData.cCount>0 )
				{
					cornerValues = AllocPointer< Real >( _oldCCount );
					if( nonLinearFit ) cornerGradients = AllocPointer< Point< Real , Dim > >( _oldCCount );
					cornerSet = AllocPointer< char >( _oldCCount );
				}
			}
			if( _oldECount<sliceData.eCount )
			{
				_oldECount = sliceData.eCount;
				FreePointer( edgeKeys ) ; FreePointer( edgeSet );
				edgeKeys = AllocPointer< _Key >( _oldECount );
				edgeSet = AllocPointer< char >( _oldECount );
			}
			if( _oldFCount<sliceData.fCount )
			{
				_oldFCount = sliceData.fCount;
				FreePointer( faceEdges ) ; FreePointer( faceSet );
				faceEdges = AllocPointer< _FaceEdges >( _oldFCount );
				faceSet = AllocPointer< char >( _oldFCount );
			}

			if( sliceData.cCount>0 ) memset( cornerSet , 0 , sizeof( char ) * sliceData.cCount );
			if( sliceData.eCount>0 ) memset(   edgeSet , 0 , sizeof( char ) * sliceData.eCount );
			if( sliceData.fCount>0 ) memset(   faceSet , 0 , sizeof( char ) * sliceData.fCount );
		}
	protected:
		node_index_type _oldCCount , _oldECount , _oldFCount;
		node_index_type _oldNCount;
	};

	///////////////////
	// _XSliceValues //
	///////////////////
	struct _XSliceValues
	{
		typename SliceData::XSliceTableData xSliceData;
		Pointer( _Key ) edgeKeys ; Pointer( char ) edgeSet;
		Pointer( _FaceEdges ) faceEdges ; Pointer( char ) faceSet;
		std::unordered_map< _Key , std::vector< _IsoEdge > , typename _Key::Hasher > faceEdgeMap;
		std::unordered_map< _Key , std::pair< node_index_type , Vertex > , typename _Key::Hasher > edgeVertexMap;
		std::unordered_map< _Key , _Key , typename _Key::Hasher > vertexPairMap;
		std::vector< std::vector< std::pair< _Key , std::pair< node_index_type , Vertex > > > > edgeVertexKeyValues;
		std::vector< std::vector< std::pair< _Key , _Key > > > vertexPairKeyValues;
		std::vector< std::vector< std::pair< _Key , std::vector< _IsoEdge > > > > faceEdgeKeyValues;

		_XSliceValues( void )
		{
			_oldECount = _oldFCount = 0;
			edgeKeys = NullPointer( _Key ) ; edgeSet = NullPointer( char );
			faceEdges = NullPointer( _FaceEdges ) ; faceSet = NullPointer( char );
			edgeVertexKeyValues.resize( ThreadPool::NumThreads() );
			vertexPairKeyValues.resize( ThreadPool::NumThreads() );
			faceEdgeKeyValues.resize( ThreadPool::NumThreads() );
		}
		~_XSliceValues( void )
		{
			_oldECount = _oldFCount = 0;
			FreePointer( edgeKeys ) ; FreePointer( edgeSet );
			FreePointer( faceEdges ) ; FreePointer( faceSet );
		}
		void setEdgeVertexMap( void )
		{
			for( node_index_type i=0 ; i<(node_index_type)edgeVertexKeyValues.size() ; i++ )
			{
				for( int j=0 ; j<edgeVertexKeyValues[i].size() ; j++ ) edgeVertexMap[ edgeVertexKeyValues[i][j].first ] = edgeVertexKeyValues[i][j].second;
				edgeVertexKeyValues[i].clear();
			}
		}
		void setVertexPairMap( void )
		{
			for( node_index_type i=0 ; i<(node_index_type)vertexPairKeyValues.size() ; i++ )
			{
				for( int j=0 ; j<vertexPairKeyValues[i].size() ; j++ )
				{
					vertexPairMap[ vertexPairKeyValues[i][j].first ] = vertexPairKeyValues[i][j].second;
					vertexPairMap[ vertexPairKeyValues[i][j].second ] = vertexPairKeyValues[i][j].first;
				}
				vertexPairKeyValues[i].clear();
			}
		}
		void setFaceEdgeMap( void )
		{
			for( node_index_type i=0 ; i<(node_index_type)faceEdgeKeyValues.size() ; i++ )
			{
				for( int j=0 ; j<faceEdgeKeyValues[i].size() ; j++ )
				{
					auto iter = faceEdgeMap.find( faceEdgeKeyValues[i][j].first );
					if( iter==faceEdgeMap.end() ) faceEdgeMap[ faceEdgeKeyValues[i][j].first ] = faceEdgeKeyValues[i][j].second;
					else for( int k=0 ; k<faceEdgeKeyValues[i][j].second.size() ; k++ ) iter->second.push_back( faceEdgeKeyValues[i][j].second[k] );
				}
				faceEdgeKeyValues[i].clear();
			}
		}
		void reset( void )
		{
			faceEdgeMap.clear() , edgeVertexMap.clear() , vertexPairMap.clear();
			for( node_index_type i=0 ; i<(node_index_type)edgeVertexKeyValues.size() ; i++ ) edgeVertexKeyValues[i].clear();
			for( node_index_type i=0 ; i<(node_index_type)vertexPairKeyValues.size() ; i++ ) vertexPairKeyValues[i].clear();
			for( node_index_type i=0 ; i<(node_index_type)faceEdgeKeyValues.size() ; i++ ) faceEdgeKeyValues[i].clear();

			if( _oldECount<xSliceData.eCount )
			{
				_oldECount = xSliceData.eCount;
				FreePointer( edgeKeys ) ; FreePointer( edgeSet );
				edgeKeys = AllocPointer< _Key >( _oldECount );
				edgeSet = AllocPointer< char >( _oldECount );
			}
			if( _oldFCount<xSliceData.fCount )
			{
				_oldFCount = xSliceData.fCount;
				FreePointer( faceEdges ) ; FreePointer( faceSet );
				faceEdges = AllocPointer< _FaceEdges >( _oldFCount );
				faceSet = AllocPointer< char >( _oldFCount );
			}
			if( xSliceData.eCount>0 ) memset( edgeSet , 0 , sizeof( char ) * xSliceData.eCount );
			if( xSliceData.fCount>0 ) memset( faceSet , 0 , sizeof( char ) * xSliceData.fCount );
		}

	protected:
		node_index_type _oldECount , _oldFCount;
	};

	/////////////////
	// _SlabValues //
	/////////////////
	struct _SlabValues
	{
	protected:
		_XSliceValues _xSliceValues[2];
		_SliceValues _sliceValues[2];
	public:
		_SliceValues& sliceValues( int idx ){ return _sliceValues[idx&1]; }
		const _SliceValues& sliceValues( int idx ) const { return _sliceValues[idx&1]; }
		_XSliceValues& xSliceValues( int idx ){ return _xSliceValues[idx&1]; }
		const _XSliceValues& xSliceValues( int idx ) const { return _xSliceValues[idx&1]; }
	};

	template< unsigned int ... FEMSigs >
	static void _SetSliceIsoCorners( const FEMTree< Dim , Real >& tree , ConstPointer( Real ) coefficients , ConstPointer( Real ) coarseCoefficients , Real isoValue , LocalDepth depth , int slice ,         std::vector< _SlabValues >& slabValues , const _Evaluator< UIntPack< FEMSigs ... > , 1 >& evaluator )
	{
		if( slice>0          ) _SetSliceIsoCorners< FEMSigs ... >( tree , coefficients , coarseCoefficients , isoValue , depth , slice , HyperCube::FRONT , slabValues , evaluator );
		if( slice<(1<<depth) ) _SetSliceIsoCorners< FEMSigs ... >( tree , coefficients , coarseCoefficients , isoValue , depth , slice , HyperCube::BACK  , slabValues , evaluator );
	}
	template< unsigned int ... FEMSigs >
	static void _SetSliceIsoCorners( const FEMTree< Dim , Real >& tree , ConstPointer( Real ) coefficients , ConstPointer( Real ) coarseCoefficients , Real isoValue , LocalDepth depth , int slice , HyperCube::Direction zDir , std::vector< _SlabValues >& slabValues , const _Evaluator< UIntPack< FEMSigs ... > , 1 >& evaluator )
	{
		static const unsigned int FEMDegrees[] = { FEMSignature< FEMSigs >::Degree ... };
		_SliceValues& sValues = slabValues[depth].sliceValues( slice );
		bool useBoundaryEvaluation = false;
		for( int d=0 ; d<Dim ; d++ ) if( FEMDegrees[d]==0 || ( FEMDegrees[d]==1 && sValues.cornerGradients ) ) useBoundaryEvaluation = true;
		std::vector< ConstPointSupportKey< UIntPack< FEMSignature< FEMSigs >::Degree ... > > > neighborKeys( ThreadPool::NumThreads() );
		std::vector< ConstCornerSupportKey< UIntPack< FEMSignature< FEMSigs >::Degree ... > > > bNeighborKeys( ThreadPool::NumThreads() );
		if( useBoundaryEvaluation ) for( size_t i=0 ; i<neighborKeys.size() ; i++ ) bNeighborKeys[i].set( tree._localToGlobal( depth ) );
		else                        for( size_t i=0 ; i<neighborKeys.size() ; i++ )  neighborKeys[i].set( tree._localToGlobal( depth ) );
		ThreadPool::Parallel_for( tree._sNodesBegin(depth,slice-(zDir==HyperCube::BACK ? 0 : 1)) , tree._sNodesEnd(depth,slice-(zDir==HyperCube::BACK ? 0 : 1)) , [&]( unsigned int thread , size_t i )
		{
			if( tree._isValidSpaceNode( tree._sNodes.treeNodes[i] ) )
			{
				Real squareValues[ HyperCube::Cube< Dim-1 >::template ElementNum< 0 >() ];
				ConstPointSupportKey< UIntPack< FEMSignature< FEMSigs >::Degree ... > >& neighborKey = neighborKeys[ thread ];
				ConstCornerSupportKey< UIntPack< FEMSignature< FEMSigs >::Degree ... > >& bNeighborKey = bNeighborKeys[ thread ];
				TreeNode* leaf = tree._sNodes.treeNodes[i];
				if( !IsActiveNode< Dim >( leaf->children ) )
				{
					const typename SliceData::SquareCornerIndices& cIndices = sValues.sliceData.cornerIndices( leaf );

					bool isInterior = tree._isInteriorlySupported( UIntPack< FEMSignature< FEMSigs >::Degree ... >() , leaf->parent );
					if( useBoundaryEvaluation ) bNeighborKey.getNeighbors( leaf );
					else                         neighborKey.getNeighbors( leaf );

					for( typename HyperCube::Cube< Dim-1 >::template Element< 0 > _c ; _c<HyperCube::Cube< Dim-1 >::template ElementNum< 0 >() ; _c++ )
					{
						typename HyperCube::Cube< Dim >::template Element< 0 > c( zDir , _c.index );
						node_index_type vIndex = cIndices[_c.index];
						if( !sValues.cornerSet[vIndex] )
						{
							if( sValues.cornerGradients )
							{
								CumulativeDerivativeValues< Real , Dim , 1 > p;
								if( useBoundaryEvaluation ) p = tree.template _getCornerValues< Real , 1 >( bNeighborKey , leaf , c.index , coefficients , coarseCoefficients , evaluator , tree._maxDepth , isInterior );
								else                        p = tree.template _getCornerValues< Real , 1 >(  neighborKey , leaf , c.index , coefficients , coarseCoefficients , evaluator , tree._maxDepth , isInterior );
								sValues.cornerValues[vIndex] = p[0] , sValues.cornerGradients[vIndex] = Point< Real , Dim >( p[1] , p[2] , p[3] );
							}
							else
							{
								if( useBoundaryEvaluation ) sValues.cornerValues[vIndex] = tree.template _getCornerValues< Real , 0 >( bNeighborKey , leaf , c.index , coefficients , coarseCoefficients , evaluator , tree._maxDepth , isInterior )[0];
								else                        sValues.cornerValues[vIndex] = tree.template _getCornerValues< Real , 0 >(  neighborKey , leaf , c.index , coefficients , coarseCoefficients , evaluator , tree._maxDepth , isInterior )[0];
							}
							sValues.cornerSet[vIndex] = 1;
						}
						squareValues[_c.index] = sValues.cornerValues[ vIndex ];
						TreeNode* node = leaf;
						LocalDepth _depth = depth;
						int _slice = slice;
						while( tree._isValidSpaceNode( node->parent ) && (node-node->parent->children)==c.index )
						{
							node = node->parent , _depth-- , _slice >>= 1;
							_SliceValues& _sValues = slabValues[_depth].sliceValues( _slice );
							const typename SliceData::SquareCornerIndices& _cIndices = _sValues.sliceData.cornerIndices( node );
							node_index_type _vIndex = _cIndices[_c.index];
							_sValues.cornerValues[_vIndex] = sValues.cornerValues[vIndex];
							if( _sValues.cornerGradients ) _sValues.cornerGradients[_vIndex] = sValues.cornerGradients[vIndex];
							_sValues.cornerSet[_vIndex] = 1;
						}
					}
					sValues.mcIndices[ i - sValues.sliceData.nodeOffset ] = HyperCube::Cube< Dim-1 >::MCIndex( squareValues , isoValue );
				}
			}
		}
		);
	}
	/////////////////
	// _VertexData //
	/////////////////
	class _VertexData
	{
	public:
		static _Key EdgeIndex( const TreeNode* node , typename HyperCube::Cube< Dim >::template Element< 1 > e , int maxDepth )
		{
			_Key key;
			const HyperCube::Direction* x = SliceData::template HyperCubeTables< Dim , 1 >::Directions[ e.index ];
			int d , off[Dim];
			node->depthAndOffset( d , off );
			for( int dd=0 ; dd<Dim ; dd++ )
			{
				if( x[dd]==HyperCube::CROSS )
				{
					key[(dd+0)%3] = (int)BinaryNode::CornerIndex( maxDepth+1 , d+1 , off[(dd+0)%3]<<1 , 1 );
					key[(dd+1)%3] = (int)BinaryNode::CornerIndex( maxDepth+1 , d   , off[(dd+1)%3] , x[(dd+1)%3]==HyperCube::BACK ? 0 : 1 );
					key[(dd+2)%3] = (int)BinaryNode::CornerIndex( maxDepth+1 , d   , off[(dd+2)%3] , x[(dd+2)%3]==HyperCube::BACK ? 0 : 1 );
				}
			}
			return key;
		}

		static _Key FaceIndex( const TreeNode* node , typename HyperCube::Cube< Dim >::template Element< Dim-1 > f , int maxDepth )
		{
			_Key key;
			const HyperCube::Direction* x = SliceData::template HyperCubeTables< Dim , 2 >::Directions[ f.index ];
			int d , o[Dim];
			node->depthAndOffset( d , o );
			for( int dd=0 ; dd<Dim ; dd++ )
				if( x[dd]==HyperCube::CROSS ) key[dd] = (int)BinaryNode::CornerIndex( maxDepth+1 , d+1 , o[dd]<<1 , 1 );
				else                          key[dd] = (int)BinaryNode::CornerIndex( maxDepth+1 , d   , o[dd]    , x[dd]==HyperCube::BACK ? 0 : 1 );
			return key;
		}
	};

	template< unsigned int WeightDegree , typename Data , unsigned int DataSig >
	static void _SetSliceIsoVertices( const FEMTree< Dim , Real >& tree , typename FEMIntegrator::template PointEvaluator< IsotropicUIntPack< Dim , DataSig > , ZeroUIntPack< Dim > >* pointEvaluator , const DensityEstimator< WeightDegree >* densityWeights , const SparseNodeData< ProjectiveData< Data , Real > , IsotropicUIntPack< Dim , DataSig > >* data , Real isoValue , LocalDepth depth , int slice , node_index_type& vOffset , CoredMeshData< Vertex , node_index_type >& mesh , std::vector< _SlabValues >& slabValues , std::function< void ( Vertex& , Point< Real , Dim > , Real , Data ) > SetVertex )
	{
		if( slice>0          ) _SetSliceIsoVertices< WeightDegree , Data , DataSig >( tree , pointEvaluator , densityWeights , data , isoValue , depth , slice , HyperCube::FRONT , vOffset , mesh , slabValues , SetVertex );
		if( slice<(1<<depth) ) _SetSliceIsoVertices< WeightDegree , Data , DataSig >( tree , pointEvaluator , densityWeights , data , isoValue , depth , slice , HyperCube::BACK  , vOffset , mesh , slabValues , SetVertex );
	}
	template< unsigned int WeightDegree , typename Data , unsigned int DataSig >
	static void _SetSliceIsoVertices( const FEMTree< Dim , Real >& tree , typename FEMIntegrator::template PointEvaluator< IsotropicUIntPack< Dim , DataSig > , ZeroUIntPack< Dim > >* pointEvaluator , const DensityEstimator< WeightDegree >* densityWeights , const SparseNodeData< ProjectiveData< Data , Real > , IsotropicUIntPack< Dim , DataSig > >* data , Real isoValue , LocalDepth depth , int slice , HyperCube::Direction zDir , node_index_type& vOffset , CoredMeshData< Vertex , node_index_type >& mesh , std::vector< _SlabValues >& slabValues , std::function< void ( Vertex& , Point< Real , Dim > , Real , Data ) > SetVertex )
	{
		static const unsigned int DataDegree = FEMSignature< DataSig >::Degree;
		_SliceValues& sValues = slabValues[depth].sliceValues( slice );
		// [WARNING] In the case Degree=2, these two keys are the same, so we don't have to maintain them separately.
		std::vector< ConstOneRingNeighborKey > neighborKeys( ThreadPool::NumThreads() );
		std::vector< ConstPointSupportKey< IsotropicUIntPack< Dim , WeightDegree > > > weightKeys( ThreadPool::NumThreads() );
		std::vector< ConstPointSupportKey< IsotropicUIntPack< Dim , DataDegree > > > dataKeys( ThreadPool::NumThreads() );
		for( size_t i=0 ; i<neighborKeys.size() ; i++ ) neighborKeys[i].set( tree._localToGlobal( depth ) ) , weightKeys[i].set( tree._localToGlobal( depth ) ) , dataKeys[i].set( tree._localToGlobal( depth ) );
		ThreadPool::Parallel_for( tree._sNodesBegin(depth,slice-(zDir==HyperCube::BACK ? 0 : 1)) , tree._sNodesEnd(depth,slice-(zDir==HyperCube::BACK ? 0 : 1)) , [&]( unsigned int thread , size_t i )
		{
			if( tree._isValidSpaceNode( tree._sNodes.treeNodes[i] ) )
			{
				ConstOneRingNeighborKey& neighborKey =  neighborKeys[ thread ];
				ConstPointSupportKey< IsotropicUIntPack< Dim , WeightDegree > >& weightKey = weightKeys[ thread ];
				ConstPointSupportKey< IsotropicUIntPack< Dim , DataDegree > >& dataKey = dataKeys[ thread ];
				TreeNode* leaf = tree._sNodes.treeNodes[i];
				if( !IsActiveNode< Dim >( leaf->children ) )
				{
					node_index_type idx = (node_index_type)i - sValues.sliceData.nodeOffset;
					const typename SliceData::SquareEdgeIndices& eIndices = sValues.sliceData.edgeIndices( leaf );
					if( HyperCube::Cube< Dim-1 >::HasMCRoots( sValues.mcIndices[idx] ) )
					{
						neighborKey.getNeighbors( leaf );
						if( densityWeights ) weightKey.getNeighbors( leaf );
						if( data ) dataKey.getNeighbors( leaf );

						for( typename HyperCube::Cube< Dim-1 >::template Element< 1 > _e ; _e<HyperCube::Cube< Dim-1 >::template ElementNum< 1 >() ; _e++ )
							if( HyperCube::Cube< 1 >::HasMCRoots( HyperCube::Cube< Dim-1 >::ElementMCIndex( _e , sValues.mcIndices[idx] ) ) )
							{
								typename HyperCube::Cube< Dim >::template Element< 1 > e( zDir , _e.index );
								node_index_type vIndex = eIndices[_e.index];
								volatile char &edgeSet = sValues.edgeSet[vIndex];
								if( !edgeSet )
								{
									Vertex vertex;
									_Key key = _VertexData::EdgeIndex( leaf , e , tree._localToGlobal( tree._maxDepth ) );
									_GetIsoVertex< WeightDegree , Data , DataSig >( tree , pointEvaluator , densityWeights , data , isoValue , weightKey , dataKey , leaf , _e , zDir , sValues , vertex , SetVertex );
									bool stillOwner = false;
									std::pair< node_index_type , Vertex > hashed_vertex;
									{
										std::lock_guard< std::mutex > lock( _pointInsertionMutex );
										if( !edgeSet )
										{
											mesh.addOutOfCorePoint( vertex );
											edgeSet = 1;
											hashed_vertex = std::pair< node_index_type , Vertex >( vOffset , vertex );
											sValues.edgeKeys[ vIndex ] = key;
											vOffset++;
											stillOwner = true;
										}
									}
									if( stillOwner ) sValues.edgeVertexKeyValues[ thread ].push_back( std::pair< _Key , std::pair< node_index_type , Vertex > >( key , hashed_vertex ) );
									if( stillOwner )
									{
										// We only need to pass the iso-vertex down if the edge it lies on is adjacent to a coarser leaf
										auto IsNeeded = [&]( unsigned int depth )
										{
											bool isNeeded = false;
											typename HyperCube::Cube< Dim >::template IncidentCubeIndex< 1 > my_ic = SliceData::template HyperCubeTables< Dim , 1 >::IncidentCube[e.index];
											for( typename HyperCube::Cube< Dim >::template IncidentCubeIndex< 1 > ic ; ic<HyperCube::Cube< Dim >::template IncidentCubeNum< 1 >() ; ic++ ) if( ic!=my_ic )
											{
												unsigned int xx = SliceData::template HyperCubeTables< Dim , 1 >::CellOffset[e.index][ic.index];
												isNeeded |= !tree._isValidSpaceNode( neighborKey.neighbors[ tree._localToGlobal( depth ) ].neighbors.data[xx] );
											}
											return isNeeded;
										};
										if( IsNeeded( depth ) )
										{
											const typename HyperCube::Cube< Dim >::template Element< Dim-1 > *f = SliceData::template HyperCubeTables< Dim , 1 , Dim-1 >::OverlapElements[e.index];
											for( int k=0 ; k<2 ; k++ )
											{
												TreeNode* node = leaf;
												LocalDepth _depth = depth;
												int _slice = slice;
												while( tree._isValidSpaceNode( node->parent ) && SliceData::template HyperCubeTables< Dim , 2 , 0 >::Overlap[f[k].index][(unsigned int)(node-node->parent->children) ] )
												{
													node = node->parent , _depth-- , _slice >>= 1;
													_SliceValues& _sValues = slabValues[_depth].sliceValues( _slice );
													_sValues.edgeVertexKeyValues[ thread ].push_back( std::pair< _Key , std::pair< node_index_type , Vertex > >( key , hashed_vertex ) );
													if( !IsNeeded( _depth ) ) break;
												}
											}
										}
									}
								}
							}
					}
				}
			}
		}
		);
	}

	////////////////////
	// Iso-Extraction //
	////////////////////
	template< unsigned int WeightDegree , typename Data , unsigned int DataSig >
	static void _SetXSliceIsoVertices( const FEMTree< Dim , Real >& tree , typename FEMIntegrator::template PointEvaluator< IsotropicUIntPack< Dim , DataSig > , ZeroUIntPack< Dim > >* pointEvaluator , const DensityEstimator< WeightDegree >* densityWeights , const SparseNodeData< ProjectiveData< Data , Real > , IsotropicUIntPack< Dim , DataSig > >* data , Real isoValue , LocalDepth depth , int slab , node_index_type &vOffset , CoredMeshData< Vertex , node_index_type >& mesh , std::vector< _SlabValues >& slabValues , std::function< void ( Vertex& , Point< Real , Dim > , Real , Data ) > SetVertex )
	{
		static const unsigned int DataDegree = FEMSignature< DataSig >::Degree;
		_SliceValues& bValues = slabValues[depth].sliceValues ( slab   );
		_SliceValues& fValues = slabValues[depth].sliceValues ( slab+1 );
		_XSliceValues& xValues = slabValues[depth].xSliceValues( slab   );

		// [WARNING] In the case Degree=2, these two keys are the same, so we don't have to maintain them separately.
		std::vector< ConstOneRingNeighborKey > neighborKeys( ThreadPool::NumThreads() );
		std::vector< ConstPointSupportKey< IsotropicUIntPack< Dim , WeightDegree > > > weightKeys( ThreadPool::NumThreads() );
		std::vector< ConstPointSupportKey< IsotropicUIntPack< Dim , DataDegree > > > dataKeys( ThreadPool::NumThreads() );
		for( size_t i=0 ; i<neighborKeys.size() ; i++ ) neighborKeys[i].set( tree._localToGlobal( depth ) ) , weightKeys[i].set( tree._localToGlobal( depth ) ) , dataKeys[i].set( tree._localToGlobal( depth ) );
		ThreadPool::Parallel_for( tree._sNodesBegin(depth,slab) , tree._sNodesEnd(depth,slab) , [&]( unsigned int thread , size_t i )
		{
			if( tree._isValidSpaceNode( tree._sNodes.treeNodes[i] ) )
			{
				ConstOneRingNeighborKey& neighborKey =  neighborKeys[ thread ];
				ConstPointSupportKey< IsotropicUIntPack< Dim , WeightDegree > >& weightKey = weightKeys[ thread ];
				ConstPointSupportKey< IsotropicUIntPack< Dim , DataDegree > >& dataKey = dataKeys[ thread ];
				TreeNode* leaf = tree._sNodes.treeNodes[i];
				if( !IsActiveNode< Dim >( leaf->children ) )
				{
					unsigned char mcIndex = ( bValues.mcIndices[ i - bValues.sliceData.nodeOffset ] ) | ( fValues.mcIndices[ i - fValues.sliceData.nodeOffset ] )<<4;
					const typename SliceData::SquareCornerIndices& eIndices = xValues.xSliceData.edgeIndices( leaf );
					if( HyperCube::Cube< Dim >::HasMCRoots( mcIndex ) )
					{
						neighborKey.getNeighbors( leaf );
						if( densityWeights ) weightKey.getNeighbors( leaf );
						if( data ) dataKey.getNeighbors( leaf );
						for( typename HyperCube::Cube< Dim-1 >::template Element< 0 > _c ; _c<HyperCube::Cube< Dim-1 >::template ElementNum< 0 >() ; _c++ )
						{
							typename HyperCube::Cube< Dim >::template Element< 1 > e( HyperCube::CROSS , _c.index );
							unsigned int _mcIndex = HyperCube::Cube< Dim >::ElementMCIndex( e , mcIndex );
							if( HyperCube::Cube< 1 >::HasMCRoots( _mcIndex ) )
							{
								node_index_type vIndex = eIndices[_c.index];
								volatile char &edgeSet = xValues.edgeSet[vIndex];
								if( !edgeSet )
								{
									Vertex vertex;
									_Key key = _VertexData::EdgeIndex( leaf , e.index , tree._localToGlobal( tree._maxDepth ) );
									_GetIsoVertex< WeightDegree , Data , DataSig >( tree , pointEvaluator , densityWeights , data , isoValue , weightKey , dataKey , leaf , _c , bValues , fValues , vertex , SetVertex );
									bool stillOwner = false;
									std::pair< node_index_type , Vertex > hashed_vertex;
									{
										std::lock_guard< std::mutex > lock( _pointInsertionMutex );
										if( !edgeSet )
										{
											mesh.addOutOfCorePoint( vertex );
											edgeSet = 1;
											hashed_vertex = std::pair< node_index_type , Vertex >( vOffset , vertex );
											xValues.edgeKeys[ vIndex ] = key;
											vOffset++;
											stillOwner = true;
										}
									}
									if( stillOwner ) xValues.edgeVertexKeyValues[ thread ].push_back( std::pair< _Key , std::pair< node_index_type , Vertex > >( key , hashed_vertex ) );
									if( stillOwner )
									{
										// We only need to pass the iso-vertex down if the edge it lies on is adjacent to a coarser leaf
										auto IsNeeded = [&]( unsigned int depth )
										{
											bool isNeeded = false;
											typename HyperCube::Cube< Dim >::template IncidentCubeIndex< 1 > my_ic = SliceData::template HyperCubeTables< Dim , 1 >::IncidentCube[e.index];
											for( typename HyperCube::Cube< Dim >::template IncidentCubeIndex< 1 > ic ; ic<HyperCube::Cube< Dim >::template IncidentCubeNum< 1 >() ; ic++ ) if( ic!=my_ic )
											{
												unsigned int xx = SliceData::template HyperCubeTables< Dim , 1 >::CellOffset[e.index][ic.index];
												isNeeded |= !tree._isValidSpaceNode( neighborKey.neighbors[ tree._localToGlobal( depth ) ].neighbors.data[xx] );
											}
											return isNeeded;
										};
										if( IsNeeded( depth ) )
										{
											const typename HyperCube::Cube< Dim >::template Element< Dim-1 > *f = SliceData::template HyperCubeTables< Dim , 1 , Dim-1 >::OverlapElements[e.index];
											for( int k=0 ; k<2 ; k++ )
											{
												TreeNode* node = leaf;
												LocalDepth _depth = depth;
												int _slab = slab;
												while( tree._isValidSpaceNode( node->parent ) && SliceData::template HyperCubeTables< Dim , 2 , 0 >::Overlap[f[k].index][(unsigned int)(node-node->parent->children) ] )
												{
													node = node->parent , _depth-- , _slab >>= 1;
													_XSliceValues& _xValues = slabValues[_depth].xSliceValues( _slab );
													_xValues.edgeVertexKeyValues[ thread ].push_back( std::pair< _Key , std::pair< node_index_type , Vertex > >( key , hashed_vertex ) );
													if( !IsNeeded( _depth ) ) break;
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
		);
	}
	static void _CopyFinerSliceIsoEdgeKeys( const FEMTree< Dim , Real >& tree , LocalDepth depth , int slice , std::vector< _SlabValues >& slabValues )
	{
		if( slice>0          ) _CopyFinerSliceIsoEdgeKeys( tree , depth , slice , HyperCube::FRONT , slabValues );
		if( slice<(1<<depth) ) _CopyFinerSliceIsoEdgeKeys( tree , depth , slice , HyperCube::BACK  , slabValues );
	}
	static void _CopyFinerSliceIsoEdgeKeys( const FEMTree< Dim , Real >& tree , LocalDepth depth , int slice , HyperCube::Direction zDir , std::vector< _SlabValues >& slabValues )
	{
		_SliceValues& pSliceValues = slabValues[depth  ].sliceValues(slice   );
		_SliceValues& cSliceValues = slabValues[depth+1].sliceValues(slice<<1);
		typename SliceData::SliceTableData& pSliceData = pSliceValues.sliceData;
		typename SliceData::SliceTableData& cSliceData = cSliceValues.sliceData;
		ThreadPool::Parallel_for( tree._sNodesBegin(depth,slice-(zDir==HyperCube::BACK ? 0 : 1)) , tree._sNodesEnd(depth,slice-(zDir==HyperCube::BACK ? 0 : 1)) , [&]( unsigned int thread , size_t i )
		{
			if( tree._isValidSpaceNode( tree._sNodes.treeNodes[i] ) ) if( IsActiveNode< Dim >( tree._sNodes.treeNodes[i]->children ) )
			{
				typename SliceData::SquareEdgeIndices& pIndices = pSliceData.edgeIndices( (node_index_type)i );
				// Copy the edges that overlap the coarser edges
				for( typename HyperCube::Cube< Dim-1 >::template Element< 1 > _e ; _e<HyperCube::Cube< Dim-1 >::template ElementNum< 1 >() ; _e++ )
				{
					node_index_type pIndex = pIndices[_e.index];
					if( !pSliceValues.edgeSet[ pIndex ] )
					{
						typename HyperCube::Cube< Dim >::template Element< 1 > e( zDir , _e.index );
						const typename HyperCube::Cube< Dim >::template Element< 0 > *c = SliceData::template HyperCubeTables< Dim , 1 , 0 >::OverlapElements[e.index];
						// [SANITY CHECK]
						//						if( tree._isValidSpaceNode( tree._sNodes.treeNodes[i]->children + c[0].index )!=tree._isValidSpaceNode( tree._sNodes.treeNodes[i]->children + c[1].index ) ) ERROR_OUT( "Finer edges should both be valid or invalid" );
						if( !tree._isValidSpaceNode( tree._sNodes.treeNodes[i]->children + c[0].index ) || !tree._isValidSpaceNode( tree._sNodes.treeNodes[i]->children + c[1].index ) ) continue;

						node_index_type cIndex1 = cSliceData.edgeIndices( tree._sNodes.treeNodes[i]->children + c[0].index )[_e.index];
						node_index_type cIndex2 = cSliceData.edgeIndices( tree._sNodes.treeNodes[i]->children + c[1].index )[_e.index];
						if( cSliceValues.edgeSet[cIndex1] != cSliceValues.edgeSet[cIndex2] )
						{
							_Key key;
							if( cSliceValues.edgeSet[cIndex1] ) key = cSliceValues.edgeKeys[cIndex1];
							else                                key = cSliceValues.edgeKeys[cIndex2];
							pSliceValues.edgeKeys[pIndex] = key;
							pSliceValues.edgeSet[pIndex] = 1;
						}
						else if( cSliceValues.edgeSet[cIndex1] && cSliceValues.edgeSet[cIndex2] )
						{
							_Key key1 = cSliceValues.edgeKeys[cIndex1] , key2 = cSliceValues.edgeKeys[cIndex2];
							pSliceValues.vertexPairKeyValues[ thread ].push_back( std::pair< _Key , _Key >( key1 , key2 ) );

							const TreeNode* node = tree._sNodes.treeNodes[i];
							LocalDepth _depth = depth;
							int _slice = slice;
							while( tree._isValidSpaceNode( node->parent ) && SliceData::template HyperCubeTables< Dim , 1 , 0 >::Overlap[e.index][(unsigned int)(node-node->parent->children) ] )
							{
								node = node->parent , _depth-- , _slice >>= 1;
								_SliceValues& _pSliceValues = slabValues[_depth].sliceValues(_slice);
								_pSliceValues.vertexPairKeyValues[ thread ].push_back( std::pair< _Key , _Key >( key1 , key2 ) );
							}
						}
					}
				}
			}
		}
		);
	}
	static void _CopyFinerXSliceIsoEdgeKeys( const FEMTree< Dim , Real >& tree , LocalDepth depth , int slab , std::vector< _SlabValues>& slabValues )
	{
		_XSliceValues& pSliceValues  = slabValues[depth  ].xSliceValues(slab);
		_XSliceValues& cSliceValues0 = slabValues[depth+1].xSliceValues( (slab<<1)|0 );
		_XSliceValues& cSliceValues1 = slabValues[depth+1].xSliceValues( (slab<<1)|1 );
		typename SliceData::XSliceTableData& pSliceData  = pSliceValues.xSliceData;
		typename SliceData::XSliceTableData& cSliceData0 = cSliceValues0.xSliceData;
		typename SliceData::XSliceTableData& cSliceData1 = cSliceValues1.xSliceData;
		ThreadPool::Parallel_for( tree._sNodesBegin(depth,slab) , tree._sNodesEnd(depth,slab) , [&]( unsigned int thread , size_t i )
		{
			if( tree._isValidSpaceNode( tree._sNodes.treeNodes[i] ) ) if( IsActiveNode< Dim >( tree._sNodes.treeNodes[i]->children ) )
			{
				typename SliceData::SquareCornerIndices& pIndices = pSliceData.edgeIndices( (node_index_type)i );
				for( typename HyperCube::Cube< Dim-1 >::template Element< 0 > _c ; _c<HyperCube::Cube< Dim-1 >::template ElementNum< 0 >() ; _c++ )
				{
					typename HyperCube::Cube< Dim >::template Element< 1 > e( HyperCube::CROSS , _c.index );
					node_index_type pIndex = pIndices[ _c.index ];
					if( !pSliceValues.edgeSet[pIndex] )
					{
						typename HyperCube::Cube< Dim >::template Element< 0 > c0( HyperCube::BACK , _c.index ) , c1( HyperCube::FRONT , _c.index );

						// [SANITY CHECK]
						//					if( tree._isValidSpaceNode( tree._sNodes.treeNodes[i]->children + c0 )!=tree._isValidSpaceNode( tree._sNodes.treeNodes[i]->children + c1 ) ) ERROR_OUT( "Finer edges should both be valid or invalid" );
						if( !tree._isValidSpaceNode( tree._sNodes.treeNodes[i]->children + c0.index ) || !tree._isValidSpaceNode( tree._sNodes.treeNodes[i]->children + c1.index ) ) continue;

						node_index_type cIndex0 = cSliceData0.edgeIndices( tree._sNodes.treeNodes[i]->children + c0.index )[_c.index];
						node_index_type cIndex1 = cSliceData1.edgeIndices( tree._sNodes.treeNodes[i]->children + c1.index )[_c.index];
						// If there's one zero-crossing along the edge
						if( cSliceValues0.edgeSet[cIndex0] != cSliceValues1.edgeSet[cIndex1] )
						{
							_Key key;
							if( cSliceValues0.edgeSet[cIndex0] ) key = cSliceValues0.edgeKeys[cIndex0]; //, vPair = cSliceValues0.edgeVertexMap.find( key )->second;
							else                                 key = cSliceValues1.edgeKeys[cIndex1]; //, vPair = cSliceValues1.edgeVertexMap.find( key )->second;
							pSliceValues.edgeKeys[ pIndex ] = key;
							pSliceValues.edgeSet[ pIndex ] = 1;
						}
						// If there's are two zero-crossings along the edge
						else if( cSliceValues0.edgeSet[cIndex0] && cSliceValues1.edgeSet[cIndex1] )
						{
							_Key key0 = cSliceValues0.edgeKeys[cIndex0] , key1 = cSliceValues1.edgeKeys[cIndex1];
							pSliceValues.vertexPairKeyValues[ thread ].push_back( std::pair< _Key , _Key >( key0 , key1 ) );
							const TreeNode* node = tree._sNodes.treeNodes[i];
							LocalDepth _depth = depth;
							int _slab = slab;
							while( tree._isValidSpaceNode( node->parent ) && SliceData::template HyperCubeTables< Dim , 1 , 0 >::Overlap[e.index][(unsigned int)(node-node->parent->children) ] )
							{
								node = node->parent , _depth-- , _slab>>= 1;
								_SliceValues& _pSliceValues = slabValues[_depth].sliceValues(_slab);
								_pSliceValues.vertexPairKeyValues[ thread ].push_back( std::pair< _Key , _Key >( key0 , key1 ) );
							}
						}
					}
				}
			}
		}
		);
	}
	static void _SetSliceIsoEdges( const FEMTree< Dim , Real >& tree , LocalDepth depth , int slice , std::vector< _SlabValues >& slabValues )
	{
		if( slice>0          ) _SetSliceIsoEdges( tree , depth , slice , HyperCube::FRONT , slabValues );
		if( slice<(1<<depth) ) _SetSliceIsoEdges( tree , depth , slice , HyperCube::BACK  , slabValues );
	}
	static void _SetSliceIsoEdges( const FEMTree< Dim , Real >& tree , LocalDepth depth , int slice , HyperCube::Direction zDir , std::vector< _SlabValues >& slabValues )
	{
		_SliceValues& sValues = slabValues[depth].sliceValues( slice );
		std::vector< ConstOneRingNeighborKey > neighborKeys( ThreadPool::NumThreads() );
		for( size_t i=0 ; i<neighborKeys.size() ; i++ ) neighborKeys[i].set( tree._localToGlobal( depth ) );
		ThreadPool::Parallel_for( tree._sNodesBegin(depth, slice-(zDir==HyperCube::BACK ? 0 : 1)) , tree._sNodesEnd(depth,slice-(zDir==HyperCube::BACK ? 0 : 1)) , [&]( unsigned int thread , size_t i )
		{
			if( tree._isValidSpaceNode( tree._sNodes.treeNodes[i] ) )
			{
				int isoEdges[ 2 * HyperCube::MarchingSquares::MAX_EDGES ];
				ConstOneRingNeighborKey& neighborKey = neighborKeys[ thread ];
				TreeNode* leaf = tree._sNodes.treeNodes[i];
				if( !IsActiveNode< Dim >( leaf->children ) )
				{
					node_index_type idx = (node_index_type)i - sValues.sliceData.nodeOffset;
					const typename SliceData::SquareEdgeIndices& eIndices = sValues.sliceData.edgeIndices( leaf );
					const typename SliceData::SquareFaceIndices& fIndices = sValues.sliceData.faceIndices( leaf );
					unsigned char mcIndex = sValues.mcIndices[idx];
					if( !sValues.faceSet[ fIndices[0] ] )
					{
						neighborKey.getNeighbors( leaf );
						unsigned int xx = WindowIndex< IsotropicUIntPack< Dim , 3 > , IsotropicUIntPack< Dim , 1 > >::Index + (zDir==HyperCube::BACK ? -1 : 1);
						if( !IsActiveNode< Dim >( neighborKey.neighbors[ tree._localToGlobal( depth ) ].neighbors.data[xx] ) || !IsActiveNode< Dim >( neighborKey.neighbors[ tree._localToGlobal( depth ) ].neighbors.data[xx]->children ) )
						{
							_FaceEdges fe;
							fe.count = HyperCube::MarchingSquares::AddEdgeIndices( mcIndex , isoEdges );
							for( int j=0 ; j<fe.count ; j++ ) for( int k=0 ; k<2 ; k++ )
							{
								if( !sValues.edgeSet[ eIndices[ isoEdges[2*j+k] ] ] ) ERROR_OUT( "Edge not set: " , slice , " / " , 1<<depth );
								fe.edges[j][k] = sValues.edgeKeys[ eIndices[ isoEdges[2*j+k] ] ];
							}
							sValues.faceSet[ fIndices[0] ] = 1;
							sValues.faceEdges[ fIndices[0] ] = fe;

							TreeNode* node = leaf;
							LocalDepth _depth = depth;
							int _slice = slice;
							typename HyperCube::Cube< Dim >::template Element< Dim-1 > f( zDir , 0 );
							std::vector< _IsoEdge > edges;
							edges.resize( fe.count );
							for( int j=0 ; j<fe.count ; j++ ) edges[j] = fe.edges[j];
							while( tree._isValidSpaceNode( node->parent ) && SliceData::template HyperCubeTables< Dim , 2 , 0 >::Overlap[f.index][(unsigned int)(node-node->parent->children) ] )
							{
								node = node->parent , _depth-- , _slice >>= 1;
								if( IsActiveNode< Dim >( neighborKey.neighbors[ tree._localToGlobal( _depth ) ].neighbors.data[xx] ) && IsActiveNode< Dim >( neighborKey.neighbors[ tree._localToGlobal( _depth ) ].neighbors.data[xx]->children ) ) break;
								_Key key = _VertexData::FaceIndex( node , f , tree._localToGlobal( tree._maxDepth ) );
								_SliceValues& _sValues = slabValues[_depth].sliceValues( _slice );
								_sValues.faceEdgeKeyValues[ thread ].push_back( std::pair< _Key , std::vector< _IsoEdge > >( key , edges ) );
							}
						}
					}
				}
			}
		}
		);
	}
	static void _SetXSliceIsoEdges( const FEMTree< Dim , Real >& tree , LocalDepth depth , int slab , std::vector< _SlabValues >& slabValues )
	{
		_SliceValues& bValues = slabValues[depth].sliceValues ( slab   );
		_SliceValues& fValues = slabValues[depth].sliceValues ( slab+1 );
		_XSliceValues& xValues = slabValues[depth].xSliceValues( slab   );

		std::vector< ConstOneRingNeighborKey > neighborKeys( ThreadPool::NumThreads() );
		for( size_t i=0 ; i<neighborKeys.size() ; i++ ) neighborKeys[i].set( tree._localToGlobal( depth ) );
		ThreadPool::Parallel_for( tree._sNodesBegin(depth,slab) , tree._sNodesEnd(depth,slab) , [&]( unsigned int thread , size_t i )
		{
			if( tree._isValidSpaceNode( tree._sNodes.treeNodes[i] ) )
			{
				int isoEdges[ 2 * HyperCube::MarchingSquares::MAX_EDGES ];
				ConstOneRingNeighborKey& neighborKey = neighborKeys[ thread ];
				TreeNode* leaf = tree._sNodes.treeNodes[i];
				if( !IsActiveNode< Dim >( leaf->children ) )
				{
					const typename SliceData::SquareCornerIndices& cIndices = xValues.xSliceData.edgeIndices( leaf );
					const typename SliceData::SquareEdgeIndices& eIndices = xValues.xSliceData.faceIndices( leaf );
					unsigned char mcIndex = ( bValues.mcIndices[ i - bValues.sliceData.nodeOffset ] ) | ( fValues.mcIndices[ i - fValues.sliceData.nodeOffset ]<<4 );
					{
						neighborKey.getNeighbors( leaf );
						// Iterate over the edges on the back
						for( typename HyperCube::Cube< Dim-1 >::template Element< 1 > _e ; _e<HyperCube::Cube< Dim-1 >::template ElementNum< 1 >() ; _e++ )
						{
							typename HyperCube::Cube< Dim >::template Element< 2 > f( HyperCube::CROSS , _e.index );
							unsigned char _mcIndex = HyperCube::Cube< Dim >::template ElementMCIndex< 2 >( f , mcIndex );

							unsigned int xx = SliceData::template HyperCubeTables< Dim , 2 >::CellOffsetAntipodal[f.index];
							if(	!xValues.faceSet[ eIndices[_e.index] ] && ( !IsActiveNode< Dim >( neighborKey.neighbors[ tree._localToGlobal( depth ) ].neighbors.data[xx] ) || !IsActiveNode< Dim >( neighborKey.neighbors[ tree._localToGlobal( depth ) ].neighbors.data[xx]->children ) ) )
							{
								_FaceEdges fe;
								fe.count = HyperCube::MarchingSquares::AddEdgeIndices( _mcIndex , isoEdges );
								for( int j=0 ; j<fe.count ; j++ ) for( int k=0 ; k<2 ; k++ )
								{
									typename HyperCube::Cube< Dim >::template Element< 1 > e( f , typename HyperCube::Cube< Dim-1 >::template Element< 1 >( isoEdges[2*j+k] ) );
									HyperCube::Direction dir ; unsigned int coIndex;
									e.factor( dir , coIndex );
									if( dir==HyperCube::CROSS ) // Cross-edge
									{
										node_index_type idx = cIndices[ coIndex ];
										if( !xValues.edgeSet[ idx ] ) ERROR_OUT( "Edge not set: " , slab , " / " , 1<<depth );
										fe.edges[j][k] = xValues.edgeKeys[ idx ];
									}
									else
									{
										const _SliceValues& sValues = dir==HyperCube::BACK ? bValues : fValues;
										node_index_type idx = sValues.sliceData.edgeIndices((node_index_type)i)[ coIndex ];
										if( !sValues.edgeSet[ idx ] ) ERROR_OUT( "Edge not set: " , slab , " / " , 1<<depth );
										fe.edges[j][k] = sValues.edgeKeys[ idx ];
									}
								}
								xValues.faceSet[ eIndices[_e.index] ] = 1;
								xValues.faceEdges[ eIndices[_e.index] ] = fe;

								TreeNode* node = leaf;
								LocalDepth _depth = depth;
								int _slab = slab;
								std::vector< _IsoEdge > edges;
								edges.resize( fe.count );
								for( int j=0 ; j<fe.count ; j++ ) edges[j] = fe.edges[j];
								while( tree._isValidSpaceNode( node->parent ) && SliceData::template HyperCubeTables< Dim , 2 , 0 >::Overlap[f.index][(unsigned int)(node-node->parent->children) ] )
								{
									node = node->parent , _depth-- , _slab >>= 1;
									if( IsActiveNode< Dim >( neighborKey.neighbors[ tree._localToGlobal( _depth ) ].neighbors.data[xx] ) && IsActiveNode< Dim >( neighborKey.neighbors[ tree._localToGlobal( _depth ) ].neighbors.data[xx]->children ) ) break;
									_Key key = _VertexData::FaceIndex( node , f , tree._localToGlobal( tree._maxDepth ) );
									_XSliceValues& _xValues = slabValues[_depth].xSliceValues( _slab );
									_xValues.faceEdgeKeyValues[ thread ].push_back( std::pair< _Key , std::vector< _IsoEdge > >( key , edges ) );
								}
							}
						}
					}
				}
			}
		}
		);
	}

	static void _SetIsoSurface( const FEMTree< Dim , Real >& tree , LocalDepth depth , int offset , const _SliceValues& bValues , const _SliceValues& fValues , const _XSliceValues& xValues , CoredMeshData< Vertex , node_index_type >& mesh , bool polygonMesh , bool addBarycenter , node_index_type& vOffset , bool flipOrientation )
	{
		std::vector< std::pair< node_index_type , Vertex > > polygon;
		std::vector< std::vector< _IsoEdge > > edgess( ThreadPool::NumThreads() );
		ThreadPool::Parallel_for( tree._sNodesBegin(depth,offset) , tree._sNodesEnd(depth,offset) , [&]( unsigned int thread , size_t i )
		{
			if( tree._isValidSpaceNode( tree._sNodes.treeNodes[i] ) )
			{
				std::vector< _IsoEdge >& edges = edgess[ thread ];
				TreeNode* leaf = tree._sNodes.treeNodes[i];
				int res = 1<<depth;
				LocalDepth d ; LocalOffset off;
				tree._localDepthAndOffset( leaf , d , off );
				bool inBounds = off[0]>=0 && off[0]<res && off[1]>=0 && off[1]<res && off[2]>=0 && off[2]<res;
				if( inBounds && !IsActiveNode< Dim >( leaf->children ) )
				{
					edges.clear();
					unsigned char mcIndex = ( bValues.mcIndices[ i - bValues.sliceData.nodeOffset ] ) | ( fValues.mcIndices[ i - fValues.sliceData.nodeOffset ]<<4 );
					// [WARNING] Just because the node looks empty doesn't mean it doesn't get eges from finer neighbors
					{
						// Gather the edges from the faces (with the correct orientation)
						for( typename HyperCube::Cube< Dim >::template Element< Dim-1 > f ; f<HyperCube::Cube< Dim >::template ElementNum< Dim-1 >() ; f++ )
						{
							int flip = HyperCube::Cube< Dim >::IsOriented( f ) ? 0 : 1;
							HyperCube::Direction fDir = f.direction();
							if( fDir==HyperCube::BACK || fDir==HyperCube::FRONT )
							{
								const _SliceValues& sValues = (fDir==HyperCube::BACK) ? bValues : fValues;
								node_index_type fIdx = sValues.sliceData.faceIndices((node_index_type)i)[0];
								if( sValues.faceSet[fIdx] )
								{
									const _FaceEdges& fe = sValues.faceEdges[ fIdx ];
									for( int j=0 ; j<fe.count ; j++ ) edges.push_back( _IsoEdge( fe.edges[j][flip] , fe.edges[j][1-flip] ) );
								}
								else
								{
									_Key key = _VertexData::FaceIndex( leaf , f , tree._localToGlobal( tree._maxDepth ) );
									typename std::unordered_map< _Key , std::vector< _IsoEdge > , typename _Key::Hasher >::const_iterator iter = sValues.faceEdgeMap.find(key);
									if( iter!=sValues.faceEdgeMap.end() )
									{
										const std::vector< _IsoEdge >& _edges = iter->second;
										for( size_t j=0 ; j<_edges.size() ; j++ ) edges.push_back( _IsoEdge( _edges[j][flip] , _edges[j][1-flip] ) );
									}
									else ERROR_OUT( "Invalid faces: " , i , "  " , fDir==HyperCube::BACK ? "back" : ( fDir==HyperCube::FRONT ? "front" : ( fDir==HyperCube::CROSS ? "cross" : "unknown" ) ) );
								}
							}
							else
							{
								node_index_type fIdx = xValues.xSliceData.faceIndices((node_index_type)i)[ f.coIndex() ];
								if( xValues.faceSet[fIdx] )
								{
									const _FaceEdges& fe = xValues.faceEdges[ fIdx ];
									for( int j=0 ; j<fe.count ; j++ ) edges.push_back( _IsoEdge( fe.edges[j][flip] , fe.edges[j][1-flip] ) );
								}
								else
								{
									_Key key = _VertexData::FaceIndex( leaf , f , tree._localToGlobal( tree._maxDepth ) );
									typename std::unordered_map< _Key , std::vector< _IsoEdge > , typename _Key::Hasher >::const_iterator iter = xValues.faceEdgeMap.find(key);
									if( iter!=xValues.faceEdgeMap.end() )
									{
										const std::vector< _IsoEdge >& _edges = iter->second;
										for( size_t j=0 ; j<_edges.size() ; j++ ) edges.push_back( _IsoEdge( _edges[j][flip] , _edges[j][1-flip] ) );
									}
									else ERROR_OUT( "Invalid faces: " , i , "  " ,  fDir==HyperCube::BACK ? "back" : ( fDir==HyperCube::FRONT ? "front" : ( fDir==HyperCube::CROSS ? "cross" : "unknown" ) ) );
								}
							}
						}
						// Get the edge loops
						std::vector< std::vector< _Key > > loops;
						while( edges.size() )
						{
							loops.resize( loops.size()+1 );
							_IsoEdge edge = edges.back();
							edges.pop_back();
							_Key start = edge[0] , current = edge[1];
							while( current!=start )
							{
								int idx;
								for( idx=0 ; idx<(int)edges.size() ; idx++ ) if( edges[idx][0]==current ) break;
								if( idx==edges.size() )
								{
									typename std::unordered_map< _Key , _Key , typename _Key::Hasher >::const_iterator iter;
									if     ( (iter=bValues.vertexPairMap.find(current))!=bValues.vertexPairMap.end() ) loops.back().push_back( current ) , current = iter->second;
									else if( (iter=fValues.vertexPairMap.find(current))!=fValues.vertexPairMap.end() ) loops.back().push_back( current ) , current = iter->second;
									else if( (iter=xValues.vertexPairMap.find(current))!=xValues.vertexPairMap.end() ) loops.back().push_back( current ) , current = iter->second;
									else
									{
										LocalDepth d ; LocalOffset off;
										tree._localDepthAndOffset( leaf , d , off );
										ERROR_OUT( "Failed to close loop [" , d-1 , ": " , off[0] , " " , off[1] , " " , off[2] , "] | (" , i , "): " , current.to_string() );
									}
								}
								else
								{
									loops.back().push_back( current );
									current = edges[idx][1];
									edges[idx] = edges.back() , edges.pop_back();
								}
							}
							loops.back().push_back( start );
						}
						// Add the loops to the mesh
						for( size_t j=0 ; j<loops.size() ; j++ )
						{
							std::vector< std::pair< node_index_type , Vertex > > polygon( loops[j].size() );
							for( size_t k=0 ; k<loops[j].size() ; k++ )
							{
								_Key key = loops[j][k];
								typename std::unordered_map< _Key , std::pair< node_index_type , Vertex > , typename _Key::Hasher >::const_iterator iter;
								size_t kk = flipOrientation ? loops[j].size()-1-k : k;
								if     ( ( iter=bValues.edgeVertexMap.find( key ) )!=bValues.edgeVertexMap.end() ) polygon[kk] = iter->second;
								else if( ( iter=fValues.edgeVertexMap.find( key ) )!=fValues.edgeVertexMap.end() ) polygon[kk] = iter->second;
								else if( ( iter=xValues.edgeVertexMap.find( key ) )!=xValues.edgeVertexMap.end() ) polygon[kk] = iter->second;
								else ERROR_OUT( "Couldn't find vertex in edge map" );
							}
							_AddIsoPolygons( thread , mesh , polygon , polygonMesh , addBarycenter , vOffset );
						}
					}
				}
			}
		}
		);
	}

	template< unsigned int WeightDegree , typename Data , unsigned int DataSig >
	static bool _GetIsoVertex( const FEMTree< Dim , Real >& tree , typename FEMIntegrator::template PointEvaluator< IsotropicUIntPack< Dim , DataSig > , ZeroUIntPack< Dim > >* pointEvaluator , const DensityEstimator< WeightDegree >* densityWeights , const SparseNodeData< ProjectiveData< Data , Real > , IsotropicUIntPack< Dim , DataSig > >* data , Real isoValue , ConstPointSupportKey< IsotropicUIntPack< Dim , WeightDegree > >& weightKey , ConstPointSupportKey< IsotropicUIntPack< Dim , FEMSignature< DataSig >::Degree > >& dataKey , const TreeNode* node , typename HyperCube::template Cube< Dim-1 >::template Element< 1 > _e , HyperCube::Direction zDir , const _SliceValues& sValues , Vertex& vertex , std::function< void ( Vertex& , Point< Real , Dim > , Real , Data ) > SetVertex )
	{
		static const unsigned int DataDegree = FEMSignature< DataSig >::Degree;
		Point< Real , Dim > position;
		int c0 , c1;
		const typename HyperCube::Cube< Dim-1 >::template Element< 0 > *_c = SliceData::template HyperCubeTables< Dim-1 , 1 , 0 >::OverlapElements[_e.index];
		c0 = _c[0].index , c1 = _c[1].index;

		bool nonLinearFit = sValues.cornerGradients!=NullPointer( Point< Real , Dim > );
		const typename SliceData::SquareCornerIndices& idx = sValues.sliceData.cornerIndices( node );
		Real x0 = sValues.cornerValues[idx[c0]] , x1 = sValues.cornerValues[idx[c1]];
		Point< Real , Dim > s;
		Real start , width;
		tree._startAndWidth( node , s , width );
		int o;
		{
			const HyperCube::Direction* dirs = SliceData::template HyperCubeTables< Dim-1 , 1 >::Directions[ _e.index ];
			for( int d=0 ; d<Dim-1 ; d++ ) if( dirs[d]==HyperCube::CROSS )
			{
				o = d;
				start = s[d];
				for( int dd=1 ; dd<Dim-1 ; dd++ ) position[(d+dd)%(Dim-1)] = s[(d+dd)%(Dim-1)] + width * ( dirs[(d+dd)%(Dim-1)]==HyperCube::BACK ? 0 : 1 );
			}
		}
		position[ Dim-1 ] = s[Dim-1] + width * ( zDir==HyperCube::BACK ? 0 : 1 );

		double averageRoot;
		bool rootFound = false;
		if( nonLinearFit )
		{
			double dx0 = sValues.cornerGradients[idx[c0]][o] * width , dx1 = sValues.cornerGradients[idx[c1]][o] * width;

			// The scaling will turn the Hermite Spline into a quadratic
			double scl = (x1-x0) / ( (dx1+dx0 ) / 2 );
			dx0 *= scl , dx1 *= scl;

			// Hermite Spline
			Polynomial< 2 > P;
			P.coefficients[0] = x0;
			P.coefficients[1] = dx0;
			P.coefficients[2] = 3*(x1-x0)-dx1-2*dx0;

			double roots[2];
			int rCount = 0 , rootCount = P.getSolutions( isoValue , roots , 0 );
			averageRoot = 0;
			for( int i=0 ; i<rootCount ; i++ ) if( roots[i]>=0 && roots[i]<=1 ) averageRoot += roots[i] , rCount++;
			if( rCount ) rootFound = true;
			averageRoot /= rCount;
		}
		if( !rootFound )
		{
			// We have a linear function L, with L(0) = x0 and L(1) = x1
			// => L(t) = x0 + t * (x1-x0)
			// => L(t) = isoValue <=> t = ( isoValue - x0 ) / ( x1 - x0 )
			if( x0==x1 ) ERROR_OUT( "Not a zero-crossing root: " , x0 , " " , x1 );
			averageRoot = ( isoValue - x0 ) / ( x1 - x0 );
		}
		if( averageRoot<=0 || averageRoot>=1 )
		{
			_BadRootCount++;
			if( averageRoot<0 ) averageRoot = 0;
			if( averageRoot>1 ) averageRoot = 1;
		}
		position[o] = Real( start + width*averageRoot );
		Real depth = (Real)1.;
		Data dataValue;
		if( densityWeights )
		{
			Real weight;
			tree._getSampleDepthAndWeight( *densityWeights , node , position , weightKey , depth , weight );
		}
		if( data )
		{
			if( DataDegree==0 ) 
			{
				Point< Real , 3 > center( s[0] + width/2 , s[1] + width/2 , s[2] + width/2 );
				dataValue = tree.template _evaluate< ProjectiveData< Data , Real > , SparseNodeData< ProjectiveData< Data , Real > , IsotropicUIntPack< Dim , DataSig > > , 0 >( *data , center , *pointEvaluator , dataKey ).value();
			}
			else dataValue = tree.template _evaluate< ProjectiveData< Data , Real > , SparseNodeData< ProjectiveData< Data , Real > , IsotropicUIntPack< Dim , DataSig > > , 0 >( *data , position , *pointEvaluator , dataKey ).value();
		}
		SetVertex( vertex , position , depth , dataValue );
		return true;
	}
	template< unsigned int WeightDegree , typename Data , unsigned int DataSig >
	static bool _GetIsoVertex( const FEMTree< Dim , Real >& tree , typename FEMIntegrator::template PointEvaluator< IsotropicUIntPack< Dim , DataSig > , ZeroUIntPack< Dim > >* pointEvaluator , const DensityEstimator< WeightDegree >* densityWeights , const SparseNodeData< ProjectiveData< Data , Real > , IsotropicUIntPack< Dim , DataSig > >* data , Real isoValue , ConstPointSupportKey< IsotropicUIntPack< Dim , WeightDegree > >& weightKey , ConstPointSupportKey< IsotropicUIntPack< Dim , FEMSignature< DataSig >::Degree > >& dataKey , const TreeNode* node , typename HyperCube::template Cube< Dim-1 >::template Element< 0 > _c , const _SliceValues& bValues , const _SliceValues& fValues , Vertex& vertex , std::function< void ( Vertex& , Point< Real , Dim > , Real , Data ) > SetVertex )
	{
		static const unsigned int DataDegree = FEMSignature< DataSig >::Degree;
		Point< Real , Dim > position;

		bool nonLinearFit = bValues.cornerGradients!=NullPointer( Point< Real , Dim > ) && fValues.cornerGradients!=NullPointer( Point< Real , Dim > );
		const typename SliceData::SquareCornerIndices& idx0 = bValues.sliceData.cornerIndices( node );
		const typename SliceData::SquareCornerIndices& idx1 = fValues.sliceData.cornerIndices( node );
		Real x0 = bValues.cornerValues[ idx0[_c.index] ] , x1 = fValues.cornerValues[ idx1[_c.index] ];
		Point< Real , Dim > s;
		Real start , width;
		tree._startAndWidth( node , s , width );
		start = s[2];
		int x , y;
		{
			const HyperCube::Direction* xx = SliceData::template HyperCubeTables< Dim-1 , 0 >::Directions[ _c.index ];
			x = xx[0]==HyperCube::BACK ? 0 : 1 , y = xx[1]==HyperCube::BACK ? 0 : 1;
		}

		position[0] = s[0] + width*x;
		position[1] = s[1] + width*y;

		double averageRoot;
		bool rootFound = false;

		if( nonLinearFit )
		{
			double dx0 = bValues.cornerGradients[ idx0[_c.index] ][2] * width , dx1 = fValues.cornerGradients[ idx1[_c.index] ][2] * width;
			// The scaling will turn the Hermite Spline into a quadratic
			double scl = (x1-x0) / ( (dx1+dx0 ) / 2 );
			dx0 *= scl , dx1 *= scl;

			// Hermite Spline
			Polynomial< 2 > P;
			P.coefficients[0] = x0;
			P.coefficients[1] = dx0;
			P.coefficients[2] = 3*(x1-x0)-dx1-2*dx0;

			double roots[2];
			int rCount = 0 , rootCount = P.getSolutions( isoValue , roots , 0 );
			averageRoot = 0;
			for( int i=0 ; i<rootCount ; i++ ) if( roots[i]>=0 && roots[i]<=1 ) averageRoot += roots[i] , rCount++;
			if( rCount ) rootFound = true;
			averageRoot /= rCount;
		}
		if( !rootFound )
		{
			// We have a linear function L, with L(0) = x0 and L(1) = x1
			// => L(t) = x0 + t * (x1-x0)
			// => L(t) = isoValue <=> t = ( isoValue - x0 ) / ( x1 - x0 )
			if( x0==x1 ) ERROR_OUT( "Not a zero-crossing root: " , x0 , " " , x1 );
			averageRoot = ( isoValue - x0 ) / ( x1 - x0 );
		}
		if( averageRoot<=0 || averageRoot>=1 )
		{
			_BadRootCount++;
		}
		position[2] = Real( start + width*averageRoot );
		Real depth = (Real)1.;
		Data dataValue;
		if( densityWeights )
		{
			Real weight;
			tree._getSampleDepthAndWeight( *densityWeights , node , position , weightKey , depth , weight );
		}
		if( data )
		{
			if( DataDegree==0 ) 
			{
				Point< Real , 3 > center( s[0] + width/2 , s[1] + width/2 , s[2] + width/2 );
				dataValue = tree.template _evaluate< ProjectiveData< Data , Real > , SparseNodeData< ProjectiveData< Data , Real > , IsotropicUIntPack< Dim , DataSig > > , 0 >( *data , center , *pointEvaluator , dataKey ).value();
			}
			else dataValue = tree.template _evaluate< ProjectiveData< Data , Real > , SparseNodeData< ProjectiveData< Data , Real > , IsotropicUIntPack< Dim , DataSig > > , 0 >( *data , position , *pointEvaluator , dataKey ).value();
		}
		SetVertex( vertex , position , depth , dataValue );
		return true;
	}

	static unsigned int _AddIsoPolygons( unsigned int thread , CoredMeshData< Vertex , node_index_type >& mesh , std::vector< std::pair< node_index_type , Vertex > >& polygon , bool polygonMesh , bool addBarycenter , node_index_type &vOffset )
	{
		if( polygonMesh )
		{
			std::vector< node_index_type > vertices( polygon.size() );
			for( unsigned int i=0 ; i<polygon.size() ; i++ ) vertices[i] = polygon[polygon.size()-1-i].first;
			mesh.addPolygon_s( thread , vertices );
			return 1;
		}
		if( polygon.size()>3 )
		{
			bool isCoplanar = false;
			std::vector< node_index_type > triangle( 3 );

			if( addBarycenter )
				for( unsigned int i=0 ; i<polygon.size() ; i++ ) for( unsigned int j=0 ; j<i ; j++ )
					if( (i+1)%polygon.size()!=j && (j+1)%polygon.size()!=i )
					{
						Vertex v1 = polygon[i].second , v2 = polygon[j].second;
						for( int k=0 ; k<3 ; k++ ) if( v1.point[k]==v2.point[k] ) isCoplanar = true;
					}
			if( isCoplanar )
			{
				Vertex c;
				c *= 0;
				for( unsigned int i=0 ; i<polygon.size() ; i++ ) c += polygon[i].second;
				c /= ( typename Vertex::Real )polygon.size();
				node_index_type cIdx;
				{
					std::lock_guard< std::mutex > lock( _pointInsertionMutex );
					cIdx = mesh.addOutOfCorePoint( c );
					vOffset++;
				}
				for( unsigned i=0 ; i<polygon.size() ; i++ )
				{
					triangle[0] = polygon[ i                  ].first;
					triangle[1] = cIdx;
					triangle[2] = polygon[(i+1)%polygon.size()].first;
					mesh.addPolygon_s( thread , triangle );
				}
				return (unsigned int)polygon.size();
			}
			else
			{
				std::vector< Point< Real , Dim > > vertices( polygon.size() );
				for( unsigned int i=0 ; i<polygon.size() ; i++ ) vertices[i] = polygon[i].second.point;
				std::vector< TriangleIndex< node_index_type > > triangles = MinimalAreaTriangulation< node_index_type , Real , Dim >( ( ConstPointer( Point< Real , Dim > ) )GetPointer( vertices ) , (node_index_type)vertices.size() );
				if( triangles.size()!=polygon.size()-2 ) ERROR_OUT( "Minimal area triangulation failed:" , triangles.size() , " != " , polygon.size()-2 );
				for( unsigned int i=0 ; i<triangles.size() ; i++ )
				{
					for( int j=0 ; j<3 ; j++ ) triangle[2-j] = polygon[ triangles[i].idx[j] ].first;
					mesh.addPolygon_s( thread , triangle );
				}
			}
		}
		else if( polygon.size()==3 )
		{
			std::vector< node_index_type > vertices( 3 );
			for( int i=0 ; i<3 ; i++ ) vertices[2-i] = polygon[i].first;
			mesh.addPolygon_s( thread , vertices );
		}
		return (unsigned int)polygon.size()-2;
	}
public:
	struct IsoStats
	{
		double cornersTime , verticesTime , edgesTime , surfaceTime;
		double copyFinerTime , setTableTime;
		IsoStats( void ) : cornersTime(0) , verticesTime(0) , edgesTime(0) , surfaceTime(0) , copyFinerTime(0) , setTableTime(0) {;}
		std::string toString( void ) const
		{
			std::stringstream stream;
			stream << "Corners / Vertices / Edges / Surface / Set Table / Copy Finer: ";
			stream << std::fixed << std::setprecision(1) << cornersTime << " / " << verticesTime << " / " << edgesTime << " / " << surfaceTime << " / " << setTableTime << " / " << copyFinerTime;
			stream << " (s)";
			return stream.str();
		}
	};
	template< typename Data , typename SetVertexFunction , unsigned int ... FEMSigs , unsigned int WeightDegree , unsigned int DataSig >
	static IsoStats Extract( UIntPack< FEMSigs ... > , UIntPack< WeightDegree > , UIntPack< DataSig > , const FEMTree< Dim , Real >& tree , const DensityEstimator< WeightDegree >* densityWeights , const SparseNodeData< ProjectiveData< Data , Real > , IsotropicUIntPack< Dim , DataSig > >* data , const DenseNodeData< Real , UIntPack< FEMSigs ... > >& coefficients , Real isoValue , CoredMeshData< Vertex , node_index_type >& mesh , const SetVertexFunction &SetVertex , bool nonLinearFit , bool addBarycenter , bool polygonMesh , bool flipOrientation )
	{
		_BadRootCount = 0u;
		IsoStats isoStats;
		static_assert( sizeof...(FEMSigs)==Dim , "[ERROR] Number of signatures should match dimension" );
		tree._setFEM1ValidityFlags( UIntPack< FEMSigs ... >() );
		static const unsigned int DataDegree = FEMSignature< DataSig >::Degree;
		static const int FEMDegrees[] = { FEMSignature< FEMSigs >::Degree ... };
		for( int d=0 ; d<Dim ; d++ ) if( FEMDegrees[d]==0 && nonLinearFit ) WARN( "Constant B-Splines do not support non-linear interpolation" ) , nonLinearFit = false;

		SliceData::SetHyperCubeTables();

		typename FEMIntegrator::template PointEvaluator< IsotropicUIntPack< Dim , DataSig > , ZeroUIntPack< Dim > >* pointEvaluator = NULL;
		if( data ) pointEvaluator = new typename FEMIntegrator::template PointEvaluator< IsotropicUIntPack< Dim , DataSig > , ZeroUIntPack< Dim > >( tree._maxDepth );
		DenseNodeData< Real , UIntPack< FEMSigs ... > > coarseCoefficients( tree._sNodesEnd( tree._maxDepth-1 ) );
		memset( coarseCoefficients() , 0 , sizeof(Real)*tree._sNodesEnd( tree._maxDepth-1 ) );
		ThreadPool::Parallel_for( tree._sNodesBegin(0) , tree._sNodesEnd( tree._maxDepth-1 ) , [&]( unsigned int, size_t i ){ coarseCoefficients[i] = coefficients[i]; } );
		typename FEMIntegrator::template RestrictionProlongation< UIntPack< FEMSigs ... > > rp;
		for( LocalDepth d=1 ; d<tree._maxDepth ; d++ ) tree._upSample( UIntPack< FEMSigs ... >() , rp , d , coarseCoefficients() );
		FEMTree< Dim , Real >::MemoryUsage();

		std::vector< _Evaluator< UIntPack< FEMSigs ... > , 1 > > evaluators( tree._maxDepth+1 );
		for( LocalDepth d=0 ; d<=tree._maxDepth ; d++ ) evaluators[d].set( tree._maxDepth );

		node_index_type vertexOffset = 0;

		std::vector< _SlabValues > slabValues( tree._maxDepth+1 );

		// Initialize the back slice
		for( LocalDepth d=tree._maxDepth ; d>=0 ; d-- )
		{
			double t = Time();
			SliceData::SetSliceTableData( tree._sNodes , &slabValues[d].sliceValues(0).sliceData , &slabValues[d].xSliceValues(0).xSliceData , &slabValues[d].sliceValues(1).sliceData , tree._localToGlobal( d ) , tree._localInset( d ) );
			isoStats.setTableTime += Time()-t;
			slabValues[d].sliceValues (0).reset( nonLinearFit );
			slabValues[d].sliceValues (1).reset( nonLinearFit );
			slabValues[d].xSliceValues(0).reset( );
		}
		for( LocalDepth d=tree._maxDepth ; d>=0 ; d-- )
		{
			// Copy edges from finer
			double t = Time();
			if( d<tree._maxDepth ) _CopyFinerSliceIsoEdgeKeys( tree , d , 0 , slabValues );
			isoStats.copyFinerTime += Time()-t , t = Time();
			_SetSliceIsoCorners< FEMSigs ... >( tree , coefficients() , coarseCoefficients() , isoValue , d , 0 , slabValues , evaluators[d] );
			isoStats.cornersTime += Time()-t , t = Time();
			_SetSliceIsoVertices< WeightDegree , Data , DataSig >( tree , pointEvaluator , densityWeights , data , isoValue , d , 0 , vertexOffset , mesh , slabValues , SetVertex );
			isoStats.verticesTime += Time()-t , t = Time();
			_SetSliceIsoEdges( tree , d , 0 , slabValues );
			isoStats.edgesTime += Time()-t , t = Time();
		}

		// Iterate over the slices at the finest level
		for( int slice=0 ; slice<( 1<<tree._maxDepth ) ; slice++ )
		{
			// Process at all depths that contain this slice
			LocalDepth d ; int o;
			for( d=tree._maxDepth , o=slice+1 ; d>=0 ; d-- , o>>=1 )
			{
				// Copy edges from finer (required to ensure we correctly track edge cancellations)
				double t = Time();
				if( d<tree._maxDepth )
				{
					_CopyFinerSliceIsoEdgeKeys( tree , d , o , slabValues );
					_CopyFinerXSliceIsoEdgeKeys( tree , d , o-1 , slabValues );
				}
				isoStats.copyFinerTime += Time()-t , t = Time();
				// Set the slice values/vertices
				_SetSliceIsoCorners< FEMSigs ... >( tree , coefficients() , coarseCoefficients() , isoValue , d , o , slabValues , evaluators[d] );
				isoStats.cornersTime += Time()-t , t = Time();
				_SetSliceIsoVertices< WeightDegree , Data , DataSig >( tree , pointEvaluator , densityWeights , data , isoValue , d , o , vertexOffset , mesh , slabValues , SetVertex );
				isoStats.verticesTime += Time()-t , t = Time();
				_SetSliceIsoEdges( tree , d , o , slabValues );
				isoStats.edgesTime += Time()-t , t = Time();

				// Set the cross-slice edges
				_SetXSliceIsoVertices< WeightDegree , Data , DataSig >( tree , pointEvaluator , densityWeights , data , isoValue , d , o-1 , vertexOffset , mesh , slabValues , SetVertex );
				isoStats.verticesTime += Time()-t , t = Time();
				_SetXSliceIsoEdges( tree , d , o-1 , slabValues );
				isoStats.edgesTime += Time()-t , t = Time();

				ThreadPool::ParallelSections
				(
					[ &slabValues , d , o ]( void ){ slabValues[d]. sliceValues(o-1).setEdgeVertexMap(); } ,
					[ &slabValues , d , o ]( void ){ slabValues[d]. sliceValues(o  ).setEdgeVertexMap(); } ,
					[ &slabValues , d , o ]( void ){ slabValues[d].xSliceValues(o-1).setEdgeVertexMap(); } ,
					[ &slabValues , d , o ]( void ){ slabValues[d]. sliceValues(o-1).setVertexPairMap(); } ,
					[ &slabValues , d , o ]( void ){ slabValues[d]. sliceValues(o  ).setVertexPairMap(); } ,
					[ &slabValues , d , o ]( void ){ slabValues[d].xSliceValues(o-1).setVertexPairMap(); } ,
					[ &slabValues , d , o ]( void ){ slabValues[d]. sliceValues(o-1).setFaceEdgeMap(); } ,
					[ &slabValues , d , o ]( void ){ slabValues[d]. sliceValues(o  ).setFaceEdgeMap(); } ,
					[ &slabValues , d , o ]( void ){ slabValues[d].xSliceValues(o-1).setFaceEdgeMap(); }
				);
				// Add the triangles
				t = Time();
				_SetIsoSurface( tree , d , o-1 , slabValues[d].sliceValues(o-1) , slabValues[d].sliceValues(o) , slabValues[d].xSliceValues(o-1) , mesh , polygonMesh , addBarycenter , vertexOffset , flipOrientation );
				isoStats.surfaceTime += Time()-t;

				if( o&1 ) break;
			}

			for( d=tree._maxDepth , o=slice+1 ; d>=0 ; d-- , o>>=1 )
			{
				// Initialize for the next pass
				if( o<(1<<(d+1)) )
				{
					double t = Time();
					SliceData::SetSliceTableData( tree._sNodes , NULL , &slabValues[d].xSliceValues(o).xSliceData , &slabValues[d].sliceValues(o+1).sliceData , tree._localToGlobal( d ) , o + tree._localInset( d ) );
					isoStats.setTableTime += Time()-t;
					slabValues[d].sliceValues(o+1).reset( nonLinearFit );
					slabValues[d].xSliceValues(o).reset();
				}
				if( o&1 ) break;
			}
		}
		FEMTree< Dim , Real >::MemoryUsage();
		if( pointEvaluator ) delete pointEvaluator;
		size_t badRootCount = _BadRootCount;
		if( badRootCount!=0 ) WARN( "bad average roots: " , badRootCount );
		return isoStats;
	}
};
template< class Real , class Vertex > std::mutex IsoSurfaceExtractor< 3 , Real , Vertex >::_pointInsertionMutex;
template< class Real , class Vertex > std::atomic< size_t > IsoSurfaceExtractor< 3 , Real , Vertex >::_BadRootCount;


template< class Real , class Vertex > template< unsigned int D , unsigned int K >
unsigned int IsoSurfaceExtractor< 3 , Real , Vertex >::SliceData::HyperCubeTables< D , K >::CellOffset[ HyperCube::Cube< D >::template ElementNum< K >() ][ HyperCube::Cube< D >::template IncidentCubeNum< K >() ];
template< class Real , class Vertex > template< unsigned int D , unsigned int K >
unsigned int IsoSurfaceExtractor< 3 , Real , Vertex >::SliceData::HyperCubeTables< D , K >::IncidentElementCoIndex[ HyperCube::Cube< D >::template ElementNum< K >() ][ HyperCube::Cube< D >::template IncidentCubeNum< K >() ];
template< class Real , class Vertex > template< unsigned int D , unsigned int K >
unsigned int IsoSurfaceExtractor< 3 , Real , Vertex >::SliceData::HyperCubeTables< D , K >::CellOffsetAntipodal[ HyperCube::Cube< D >::template ElementNum< K >() ];
template< class Real , class Vertex > template< unsigned int D , unsigned int K >
typename HyperCube::Cube< D >::template IncidentCubeIndex < K > IsoSurfaceExtractor< 3 , Real , Vertex >::SliceData::HyperCubeTables< D , K >::IncidentCube[ HyperCube::Cube< D >::template ElementNum< K >() ];
template< class Real , class Vertex > template< unsigned int D , unsigned int K >
typename HyperCube::Direction IsoSurfaceExtractor< 3 , Real , Vertex >::SliceData::HyperCubeTables< D , K >::Directions[ HyperCube::Cube< D >::template ElementNum< K >() ][ D ];
template< class Real , class Vertex > template< unsigned int D , unsigned int K1 , unsigned int K2 >
typename HyperCube::Cube< D >::template Element< K2 > IsoSurfaceExtractor< 3 , Real , Vertex >::SliceData::HyperCubeTables< D , K1 , K2 >::OverlapElements[ HyperCube::Cube< D >::template ElementNum< K1 >() ][ HyperCube::Cube< D >::template OverlapElementNum< K1 , K2 >() ];
template< class Real , class Vertex > template< unsigned int D , unsigned int K1 , unsigned int K2 >
bool IsoSurfaceExtractor< 3 , Real , Vertex >::SliceData::HyperCubeTables< D , K1 , K2 >::Overlap[ HyperCube::Cube< D >::template ElementNum< K1 >() ][ HyperCube::Cube< D >::template ElementNum< K2 >() ];
