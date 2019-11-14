/*

The interface routines for reading and writing PLY polygon files.

Greg Turk, February 1994

---------------------------------------------------------------

A PLY file contains a single polygonal _object_.

An object is composed of lists of _elements_.  Typical elements are
vertices, faces, edges and materials.

Each type of element for a given object has one or more _properties_
associated with the element type.  For instance, a vertex element may
have as properties the floating-point values x,y,z and the three unsigned
chars representing red, green and blue.

---------------------------------------------------------------

Copyright (c) 1994 The Board of Trustees of The Leland Stanford
Junior University.  All rights reserved.   

Permission to use, copy, modify and distribute this software and its   
documentation for any purpose is hereby granted without fee, provided   
that the above copyright notice and this permission notice appear in   
all copies of this software and that you do not sell the software.   

THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND,   
EXPRESS, IMPLIED OR OTHERWISE, INCLUDING WITHOUT LIMITATION, ANY   
WARRANTY OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.   
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "PlyFile.h"
#include "MyMiscellany.h"

const char *type_names[] =
{
	"invalid",
	"char",
	"short",
	"int",
	"longlong",
	"uchar",
	"ushort",
	"uint",
	"ulonglong",
	"float",
	"double",

	"int8",       // character                 1
	"uint8",      // unsigned character        1
	"int16",      // short integer             2
	"uint16",     // unsigned short integer    2
	"int32",      // integer                   4
	"uint32",     // unsigned integer          4
	"int64",      // integer                   8
	"uint64",     // unsigned integer          8
	"float32",    // single-precision float    4
	"float64",    // double-precision float    8
};

int ply_type_size[] =
{
	0,
	1,
	2,
	4,
	8,
	1,
	2,
	4,
	8,
	4,
	8,
	1,
	1,
	2,
	2,
	4,
	4,
	8,
	8,
	4,
	8
};

typedef union
{
	int  int_value;
	char byte_values[sizeof(int)];
} endian_test_type;


static int native_binary_type = -1;
static int types_checked = 0;

#define NO_OTHER_PROPS  -1

#define DONT_STORE_PROP  0
#define STORE_PROP       1

#define OTHER_PROP       0
#define NAMED_PROP       1


/* write to a file the word describing a PLY file data type */
void write_scalar_type( FILE * , int );

/* read a line from a file and break it up into separate words */
std::vector< std::string > get_words( FILE * , char ** );

/* write to a file the word describing a PLY file data type */
void write_scalar_type( FILE * , int );

/* write an item to a file */
void write_binary_item( FILE * , int , int , unsigned int , long long , unsigned long long , double , int );
void write_ascii_item ( FILE * ,       int , unsigned int , long long , unsigned long long , double , int );

/* store a value into where a pointer and a type specify */
void store_item( void * , int , int , unsigned int , long long , unsigned long long , double );

/* return the value of a stored item */
void get_stored_item( void * , int , int & , unsigned int & , long long & , unsigned long long & , double & );

/* return the value stored in an item, given ptr to it and its type */
double get_item_value( const void * , int );

/* get binary or ascii item and store it according to ptr and type */
void get_ascii_item( const std::string & , int , int & , unsigned int & , long long & , unsigned long long & , double & );
void get_binary_item( FILE * , int       , int , int & , unsigned int & , long long & , unsigned long long & , double & );

/* byte ordering */
void get_native_binary_type();
void swap_bytes( void * , int );

void check_types();

/*************/
/*  Writing  */
/*************/


/******************************************************************************
Given a file pointer, get ready to write PLY data to the file.

Entry:
fp         - the given file pointer
nelems     - number of elements in object
elem_names - list of element names
file_type  - file type, either ascii or binary

Exit:
returns a pointer to a PlyFile, used to refer to this file, or NULL if error
******************************************************************************/

PlyFile *PlyFile::_Write( FILE *fp , const std::vector< std::string > &elem_names , int file_type )
{
	/* check for NULL file pointer */
	if( fp==NULL ) return NULL;

	if( native_binary_type==-1 ) get_native_binary_type();
	if( !types_checked ) check_types();

	/* create a record for this object */

	PlyFile *plyfile = new PlyFile( fp );
	if( file_type==PLY_BINARY_NATIVE ) plyfile->file_type = native_binary_type;
	else                               plyfile->file_type = file_type;

	/* tuck aside the names of the elements */
	plyfile->elems.resize( elem_names.size() );
	for( int i=0 ; i<elem_names.size() ; i++ )
	{
		plyfile->elems[i].name = elem_names[i];
		plyfile->elems[i].num =  0;
	}

	/* return pointer to the file descriptor */
	return plyfile;
}


/******************************************************************************
Open a polygon file for writing.

Entry:
filename   - name of file to read from
nelems     - number of elements in object
elem_names - list of element names
file_type  - file type, either ascii or binary

Exit:
version - version number of PLY file
returns a file identifier, used to refer to this file, or NULL if error
******************************************************************************/

PlyFile *PlyFile::Write( const std::string &filename , const std::vector< std::string > &elem_names , int file_type , float &version )
{
	/* tack on the extension .ply, if necessary */
	std::string name = filename;
	if( name.length()<4 || name.substr( name.length()-4 )!=".ply" ) name += ".ply";

	/* open the file for writing */
	FILE *fp = fopen( name.c_str() , "wb" );
	if( fp==NULL ) return NULL;

	/* create the actual PlyFile structure */
	PlyFile *plyfile = _Write( fp , elem_names , file_type );

	/* say what PLY file version number we're writing */
	version = plyfile->version;

	/* return pointer to the file descriptor */
	return plyfile;
}


/******************************************************************************
Describe an element, including its properties and how many will be written
to the file.

Entry:
elem_name - name of element that information is being specified about
nelems    - number of elements of this type to be written
nprops    - number of properties contained in the element
prop_list - list of properties
******************************************************************************/

void PlyFile::describe_element( const std::string &elem_name , size_t nelems , int nprops , const PlyProperty *prop_list )
{
	/* look for appropriate element */
	PlyElement *elem = find_element( elem_name );
	if( elem==NULL ) ERROR_OUT( "Can't find element '" , elem_name , "'" );

	elem->num = nelems;

	/* copy the list of properties */
	elem->props.resize( nprops );
	for( int i=0 ; i<nprops ; i++ ) elem->props[i] = PlyStoredProperty( prop_list[i] , NAMED_PROP );
}


/******************************************************************************
Describe a property of an element.

Entry:
elem_name - name of element that information is being specified about
prop      - the new property
******************************************************************************/

void PlyFile::describe_property( const std::string &elem_name , const PlyProperty *prop )
{
	/* look for appropriate element */
	PlyElement *elem = find_element( elem_name );
	if( elem == NULL )
	{
		WARN( "Can't find element '" , elem_name , "'" );
		return;
	}

	elem->props.push_back( PlyStoredProperty( *prop , NAMED_PROP ) );
}


/******************************************************************************
Describe what the "other" properties are that are to be stored, and where
they are in an element.
******************************************************************************/

void PlyFile::describe_other_properties( const PlyOtherProp &other , int offset )
{
	/* look for appropriate element */
	PlyElement *elem = find_element( other.name );
	if( elem==NULL )
	{
		WARN( "Can't find element '" , other.name , "'" );
		return;
	}

	elem->props.reserve( elem->props.size() + other.props.size() );
	for( int i=0 ; i<other.props.size() ; i++ ) elem->props.push_back( PlyStoredProperty( other.props[i] , OTHER_PROP ) );

	/* save other info about other properties */
	elem->other_size = other.size;
	elem->other_offset = offset;
}


/******************************************************************************
State how many of a given element will be written.

Entry:
elem_name - name of element that information is being specified about
nelems    - number of elements of this type to be written
******************************************************************************/
void PlyFile::element_count( const std::string &elem_name , size_t nelems )
{
	/* look for appropriate element */
	PlyElement *elem = find_element( elem_name );
	if( elem==NULL ) ERROR_OUT( "Can't find element '" , elem_name , "'" );

	elem->num = nelems;
}


/******************************************************************************
Signal that we've described everything a PLY file's header and that the
header should be written to the file.
******************************************************************************/

void PlyFile::header_complete( void )
{
	fprintf( fp , "ply\n" );
	switch( file_type )
	{
	case PLY_ASCII: fprintf( fp , "format ascii 1.0\n" )                    ; break;
	case PLY_BINARY_BE: fprintf( fp , "format binary_big_endian 1.0\n" )    ; break;
	case PLY_BINARY_LE: fprintf( fp , "format binary_little_endian 1.0\n" ) ; break;
	default: ERROR_OUT( "Bad file type: " , file_type );
	}

	/* write out the comments */
	for( int i=0 ; i<comments.size() ; i++ ) fprintf( fp , "comment %s\n" , comments[i].c_str() );

	/* write out object information */
	for( int i=0 ; i<obj_info.size() ; i++ ) fprintf( fp , "obj_info %s\n" , obj_info[i].c_str() );

	/* write out information about each element */
	for( int i=0 ; i<elems.size() ; i++ )
	{
		fprintf( fp , "element %s %llu\n" , elems[i].name.c_str() , (unsigned long long)elems[i].num );

		for( int j=0 ; j<elems[i].props.size() ; j++ )
		{
			if( elems[i].props[j].prop.is_list )
			{
				fprintf( fp , "property list " );
				write_scalar_type( fp , elems[i].props[j].prop.count_external );
				fprintf( fp , " " );
				write_scalar_type( fp , elems[i].props[j].prop.external_type );
				fprintf( fp , " %s\n", elems[i].props[j].prop.name.c_str() );
			}
			else
			{
				fprintf( fp , "property " );
				write_scalar_type( fp , elems[i].props[j].prop.external_type );
				fprintf( fp , " %s\n", elems[i].props[j].prop.name.c_str() );
			}
		}
	}

	fprintf( fp , "end_header\n" );
}


/******************************************************************************
Specify which elements are going to be written.  This should be called
before a call to the routine ply_put_element().

Entry:
elem_name - name of element we're talking about
******************************************************************************/

void PlyFile::put_element_setup( const std::string &elem_name )
{
	PlyElement *elem = find_element( elem_name );
	if( elem==NULL ) ERROR_OUT( "Can't find element '" , elem_name , "'" );
	which_elem = elem;
}


/******************************************************************************
Write an element to the file.  This routine assumes that we're
writing the type of element specified in the last call to the routine
ply_put_element_setup().

Entry:
elem_ptr - pointer to the element
******************************************************************************/

void PlyFile::put_element( void *elem_ptr )
{
	char *elem_data,*item;
	char **item_ptr;
	int list_count;
	int item_size;
	int int_val;
	unsigned int uint_val;
	long long longlong_val;
	unsigned long long ulonglong_val;
	double double_val;
	char **other_ptr;

	PlyElement *elem = which_elem;
	elem_data = (char *)elem_ptr;
	other_ptr = (char **) (((char *) elem_ptr) + elem->other_offset);

	/* write out either to an ascii or binary file */

	if( file_type==PLY_ASCII )	/* write an ascii file */
	{
		/* write out each property of the element */
		for( int j=0 ; j<elem->props.size() ; j++ )
		{
			if( elem->props[j].store==OTHER_PROP ) elem_data = *other_ptr;
			else                                   elem_data = (char *)elem_ptr;
			if( elem->props[j].prop.is_list )
			{
				item = elem_data + elem->props[j].prop.count_offset;
				get_stored_item( (void *)item , elem->props[j].prop.count_internal , int_val , uint_val , longlong_val , ulonglong_val , double_val );
				write_ascii_item( fp , int_val , uint_val , longlong_val , ulonglong_val , double_val , elem->props[j].prop.count_external );
				list_count = uint_val;
				item_ptr = (char **)( elem_data + elem->props[j].prop.offset );
				item = item_ptr[0];
				item_size = ply_type_size[ elem->props[j].prop.internal_type ];
				for( int k=0 ; k<list_count ; k++ )
				{
					get_stored_item( (void *)item , elem->props[j].prop.internal_type , int_val , uint_val , longlong_val , ulonglong_val , double_val );
					write_ascii_item( fp , int_val , uint_val , longlong_val , ulonglong_val , double_val , elem->props[j].prop.external_type );
					item += item_size;
				}
			}
			else
			{
				item = elem_data + elem->props[j].prop.offset;
				get_stored_item( (void *)item , elem->props[j].prop.internal_type , int_val , uint_val , longlong_val , ulonglong_val , double_val );
				write_ascii_item( fp , int_val , uint_val , longlong_val , ulonglong_val , double_val , elem->props[j].prop.external_type );
			}
		}
		fprintf( fp , "\n" );
	}
	else		/* write a binary file */
	{
		/* write out each property of the element */
		for( int j=0 ; j<elem->props.size() ; j++ )
		{
			if (elem->props[j].store==OTHER_PROP ) elem_data = *other_ptr;
			else                                   elem_data = (char *)elem_ptr;
			if( elem->props[j].prop.is_list )
			{
				item = elem_data + elem->props[j].prop.count_offset;
				item_size = ply_type_size[ elem->props[j].prop.count_internal ];
				get_stored_item( (void *)item , elem->props[j].prop.count_internal , int_val , uint_val , longlong_val , ulonglong_val , double_val );
				write_binary_item( fp , file_type , int_val , uint_val , longlong_val , ulonglong_val , double_val , elem->props[j].prop.count_external );
				list_count = uint_val;
				item_ptr = (char **)( elem_data + elem->props[j].prop.offset );
				item = item_ptr[0];
				item_size = ply_type_size[ elem->props[j].prop.internal_type ];
				for( int k=0 ; k<list_count ; k++ )
				{
					get_stored_item( (void *)item , elem->props[j].prop.internal_type , int_val , uint_val , longlong_val , ulonglong_val , double_val );
					write_binary_item( fp , file_type , int_val , uint_val , longlong_val , ulonglong_val , double_val , elem->props[j].prop.external_type );
					item += item_size;
				}
			}
			else
			{
				item = elem_data + elem->props[j].prop.offset;
				item_size = ply_type_size[ elem->props[j].prop.internal_type ];
				get_stored_item( (void *)item , elem->props[j].prop.internal_type , int_val , uint_val , longlong_val , ulonglong_val , double_val );
				write_binary_item( fp , file_type , int_val , uint_val , longlong_val , ulonglong_val , double_val , elem->props[j].prop.external_type );
			}
		}
	}
}


/******************************************************************************
Specify a comment that will be written in the header.

Entry:
comment - the comment to be written
******************************************************************************/

void PlyFile::put_comment( const std::string &comment ){ comments.push_back( comment ); }


/******************************************************************************
Specify a piece of object information (arbitrary text) that will be written
in the header.

Entry:
obj_info - the text information to be written
******************************************************************************/

void PlyFile::put_obj_info( const std::string &obj_info ){ this->obj_info.push_back( obj_info ); }


/*************/
/*  Reading  */
/*************/



/******************************************************************************
Given a file pointer, get ready to read PLY data from the file.

Entry:
fp - the given file pointer

Exit:
nelems     - number of elements in object
elem_names - list of element names
returns a pointer to a PlyFile, used to refer to this file, or NULL if error
******************************************************************************/

PlyFile *PlyFile::_Read( FILE *fp , std::vector< std::string > &elem_names )
{
	char *orig_line;
	/* check for NULL file pointer */
	if( fp==NULL ) return NULL;

	if( native_binary_type==-1 ) get_native_binary_type();
	if( !types_checked ) check_types();

	/* create record for this object */
	std::vector< std::string > words;
	PlyFile *plyfile = new PlyFile( fp );

	/* read and parse the file's header */
	words = get_words( plyfile->fp , &orig_line );
	if( !words.size() || words[0]!="ply" ) return NULL;
	while( words.size() )
	{
		/* parse words */
		if( words[0]=="format" )
		{
			if( words.size()!=3 ) return NULL;
			if     ( words[1]=="ascii"                ) plyfile->file_type = PLY_ASCII;
			else if( words[1]=="binary_big_endian"    ) plyfile->file_type = PLY_BINARY_BE;
			else if( words[1]=="binary_little_endian" ) plyfile->file_type = PLY_BINARY_LE;
			else return NULL;
			plyfile->version = (float)atof( words[2].c_str() );
		}
		else if( words[0]=="element"    ) plyfile->add_element ( words );
		else if( words[0]=="property"   ) plyfile->add_property( words );
		else if( words[0]=="comment"    ) plyfile->add_comment ( orig_line );
		else if( words[0]=="obj_info"   ) plyfile->add_obj_info( orig_line );
		else if( words[0]=="end_header" ) break;

		words = get_words( plyfile->fp , &orig_line );
	}

	/* create tags for each property of each element, to be used */
	/* later to say whether or not to store each property for the user */
	for( int i=0 ; i<plyfile->elems.size() ; i++ )
	{
		for( int j=0 ; j<plyfile->elems[i].props.size() ; j++ ) plyfile->elems[i].props[j].store = DONT_STORE_PROP;
		plyfile->elems[i].other_offset = NO_OTHER_PROPS; /* no "other" props by default */
	}

	/* set return values about the elements */
	elem_names.resize( plyfile->elems.size() );
	for( int i=0 ; i<elem_names.size() ; i++ ) elem_names[i] = plyfile->elems[i].name;

	/* return a pointer to the file's information */
	return plyfile;
}


/******************************************************************************
Open a polygon file for reading.

Entry:
filename - name of file to read from

Exit:
nelems     - number of elements in object
elem_names - list of element names
file_type  - file type, either ascii or binary
version    - version number of PLY file
returns a file identifier, used to refer to this file, or NULL if error
******************************************************************************/

PlyFile *PlyFile::Read( const std::string &filename , std::vector< std::string > &elem_names , int &file_type , float &version )
{
	/* tack on the extension .ply, if necessary */
	std::string name = filename;
	if( name.length()<4 || name.substr( name.length()-4 )!=".ply" ) name += ".ply";

	/* open the file for reading */
	FILE *fp = fopen( name.c_str() , "rb" );
	if( fp==NULL ) return NULL;

	/* create the PlyFile data structure */
	PlyFile *plyfile = _Read( fp , elem_names );

	/* determine the file type and version */
	file_type = plyfile->file_type;
	version = plyfile->version;

	/* return a pointer to the file's information */
	return plyfile;
}


/******************************************************************************
Get information about a particular element.

Entry:
elem_name - name of element to get information about

Exit:
nelems   - number of elements of this type in the file
nprops   - number of properties
returns a list of properties, or NULL if the file doesn't contain that elem
******************************************************************************/

std::vector< PlyProperty * > PlyFile::get_element_description( const std::string &elem_name , size_t &nelems )
{
	std::vector< PlyProperty * > prop_list;

	/* find information about the element */
	PlyElement *elem = find_element( elem_name );
	if( elem==NULL ) return prop_list;
	nelems = elem->num;

	/* make a copy of the element's property list */
	prop_list.resize( elem->props.size() );
	for( int i=0 ; i<elem->props.size() ; i++ ) prop_list[i] = new PlyProperty( elem->props[i].prop );

	/* return this duplicate property list */
	return prop_list;
}

/******************************************************************************
Specify which properties of an element are to be returned.  This should be
called before a call to the routine ply_get_element().

Entry:
elem_name - which element we're talking about
nprops    - number of properties
prop_list - list of properties
******************************************************************************/

void PlyFile::get_element_setup( const std::string &elem_name , int nprops , PlyProperty *prop_list )
{
	/* find information about the element */
	PlyElement *elem = find_element( elem_name );
	which_elem = elem;

	/* deposit the property information into the element's description */
	for( int i=0 ; i<nprops ; i++ )
	{
		/* look for actual property */
		int index;
		PlyProperty *prop = elem->find_property( prop_list[i].name , index );
		if( prop==NULL )
		{
			WARN( "Can't find property '" , prop_list[i].name , "' in element '" , elem_name , "'" );
			continue;
		}

		/* store its description */
		prop->internal_type = prop_list[i].internal_type;
		prop->offset = prop_list[i].offset;
		prop->count_internal = prop_list[i].count_internal;
		prop->count_offset = prop_list[i].count_offset;

		/* specify that the user wants this property */
		elem->props[index].store = STORE_PROP;
	}
}


/******************************************************************************
Specify a property of an element that is to be returned.  This should be
called (usually multiple times) before a call to the routine ply_get_element().
This routine should be used in preference to the less flexible old routine
called ply_get_element_setup().

Entry:
elem_name - which element we're talking about
prop      - property to add to those that will be returned
******************************************************************************/

int PlyFile::get_property( const std::string &elem_name , const PlyProperty *prop )
{
	/* find information about the element */
	PlyElement *elem = find_element( elem_name );
	which_elem = elem;

	/* deposit the property information into the element's description */
	int index;
	PlyProperty *prop_ptr = elem->find_property( prop->name , index );
	if( prop_ptr==NULL ) return 0;
	prop_ptr->internal_type  = prop->internal_type;
	prop_ptr->offset         = prop->offset;
	prop_ptr->count_internal = prop->count_internal;
	prop_ptr->count_offset   = prop->count_offset;

	/* specify that the user wants this property */
	elem->props[index].store = STORE_PROP;

	return 1;
}


/******************************************************************************
Read one element from the file.  This routine assumes that we're reading
the type of element specified in the last call to the routine
ply_get_element_setup().

Entry:
elem_ptr - pointer to location where the element information should be put
******************************************************************************/

void PlyFile::get_element( void *elem_ptr )
{
	if( file_type==PLY_ASCII ) _ascii_get_element( elem_ptr );
	else                      _binary_get_element( elem_ptr );
}

/******************************************************************************
Extract the comments from the header information of a PLY file.

Exit:
num_comments - number of comments returned
returns a pointer to a list of comments
******************************************************************************/

std::vector< std::string > &PlyFile::get_comments( void ){ return comments; }

/******************************************************************************
Extract the object information (arbitrary text) from the header information
of a PLY file.

Exit:
num_obj_info - number of lines of text information returned
returns a pointer to a list of object info lines
******************************************************************************/
std::vector< std::string > &PlyFile::get_obj_info( void ){ return obj_info; }

/******************************************************************************
Make ready for "other" properties of an element-- those properties that
the user has not explicitly asked for, but that are to be stashed away
in a special structure to be carried along with the element's other
information.

Entry:
elem    - element for which we want to save away other properties
******************************************************************************/

void setup_other_props( PlyElement *elem )
{
	int size = 0;

	/* Examine each property in decreasing order of size. */
	/* We do this so that all data types will be aligned by */
	/* word, half-word, or whatever within the structure. */

	for( int type_size=8 ; type_size>0 ; type_size/=2 )
	{

		/* add up the space taken by each property, and save this information */
		/* away in the property descriptor */
		for( int i=0 ; i<elem->props.size() ; i++ )
		{
			/* don't bother with properties we've been asked to store explicitly */
			if( elem->props[i].store ) continue;
			PlyProperty &prop = elem->props[i].prop;

			/* internal types will be same as external */
			prop.internal_type = prop.external_type;
			prop.count_internal = prop.count_external;

			/* check list case */
			if( prop.is_list )
			{
				/* pointer to list */
				if( type_size==sizeof(void *) )
				{
					prop.offset = size;
					size += sizeof( void * );    /* always use size of a pointer here */
				}

				/* count of number of list elements */
				if( type_size==ply_type_size[ prop.count_external ] )
				{
					prop.count_offset = size;
					size += ply_type_size[ prop.count_external ];
				}
			}
			/* not list */
			else if( type_size==ply_type_size[ prop.external_type ] )
			{
				prop.offset = size;
				size += ply_type_size[ prop.external_type ];
			}
		}
	}

	/* save the size for the other_props structure */
	elem->other_size = size;
}


/******************************************************************************
Specify that we want the "other" properties of an element to be tucked
away within the user's structure.  The user needn't be concerned for how
these properties are stored.

Entry:
elem_name - name of element that we want to store other_props in
offset    - offset to where other_props will be stored inside user's structure

Exit:
returns pointer to structure containing description of other_props
******************************************************************************/

bool PlyFile::set_other_properties( const std::string &elem_name , int offset , PlyOtherProp &other )
{
	/* find information about the element */
	PlyElement *elem = find_element( elem_name );
	if( elem==NULL )
	{
		WARN( "Can't find element '" , elem_name , "'" );
		return false;
	}

	/* remember that this is the "current" element */
	which_elem = elem;

	/* save the offset to where to store the other_props */
	elem->other_offset = offset;

	/* place the appropriate pointers, etc. in the element's property list */
	setup_other_props( elem );

	/* create structure for describing other_props */
	other.size = elem->other_size;
	other.props.reserve( elem->props.size() );
	for( int i=0 ; i<elem->props.size() ; i++ ) if( !elem->props[i].store ) other.props.push_back( elem->props[i].prop );

	/* set other_offset pointer appropriately if there are NO other properties */
	if( !other.props.size() ) elem->other_offset = NO_OTHER_PROPS;
	return true;
}

/*************************/
/*  Other Element Stuff  */
/*************************/




/******************************************************************************
Grab all the data for an element that a user does not want to explicitly
read in.

Entry:
elem_name  - name of element whose data is to be read in
elem_count - number of instances of this element stored in the file

Exit:
returns pointer to ALL the "other" element data for this PLY file
******************************************************************************/

PlyOtherElems *PlyFile::get_other_element( std::string &elem_name , size_t elem_count )
{
	/* look for appropriate element */
	PlyElement *elem = find_element( elem_name );
	if( elem==NULL ) ERROR_OUT( "Can't find element '" , elem_name , "'" );

	if( other_elems==NULL ) other_elems = new PlyOtherElems();
	other_elems->other_list.resize( other_elems->other_list.size()+1 );
	OtherElem *other = &other_elems->other_list.back();

	/* save name of element */
	other->elem_name = elem_name;

	/* create a list to hold all the current elements */
	other->other_data.resize( elem_count );

	/* set up for getting elements */
	set_other_properties( elem_name , offsetof( OtherData , other_props ) , other->other_props );

	/* grab all these elements */
	for( int i=0 ; i<other->other_data.size() ; i++ )
	{
		/* grab and element from the file */
		get_element( (void *)&other->other_data[i] );
	}

	/* return pointer to the other elements data */
	return other_elems;
}


/******************************************************************************
Pass along a pointer to "other" elements that we want to save in a given
PLY file.  These other elements were presumably read from another PLY file.

Entry:
other_elems - info about other elements that we want to store
******************************************************************************/

void PlyFile::describe_other_elements( PlyOtherElems *other_elems )
{
	/* ignore this call if there is no other element */
	if( other_elems==NULL ) return;

	/* save pointer to this information */
	this->other_elems = other_elems;

	/* describe the other properties of this element */
	/* store them in the main element list as elements with
	only other properties */

	elems.reserve( elems.size() + other_elems->other_list.size() );
	for( int i=0 ; i<other_elems->other_list.size() ; i++ )
	{
		PlyElement elem;
		elem.name = other_elems->other_list[i].elem_name;
		elem.num = other_elems->other_list[i].other_data.size();
		elem.props.resize(0);
		describe_other_properties( other_elems->other_list[i].other_props , offsetof( OtherData , other_props ) );
		elems.push_back( elem );
	}
}


/******************************************************************************
Write out the "other" elements specified for this PLY file.
******************************************************************************/

void PlyFile::put_other_elements( void )
{
	OtherElem *other;

	/* make sure we have other elements to write */
	if( other_elems==NULL ) return;

	/* write out the data for each "other" element */
	for( int i=0 ; i<other_elems->other_list.size() ; i++ )
	{
		other = &(other_elems->other_list[i]);
		put_element_setup( other->elem_name );

		/* write out each instance of the current element */
		for( int j=0 ; j<other->other_data.size() ; j++ ) put_element( (void *)&other->other_data[j] );
	}
}

/*******************/
/*  Miscellaneous  */
/*******************/

/******************************************************************************
Get version number and file type of a PlyFile.

Exit:
version - version of the file
file_type - PLY_ASCII, PLY_BINARY_BE, or PLY_BINARY_LE
******************************************************************************/

void PlyFile::get_info( float &version, int &file_type ){ version = this->version , file_type = this->file_type; }

/******************************************************************************
Find an element from the element list of a given PLY object.

Entry:
element - name of element we're looking for

Exit:
returns the element, or NULL if not found
******************************************************************************/

PlyElement *PlyFile::find_element( const std::string &element )
{
	for( int i=0 ; i<elems.size() ; i++ ) if( element==elems[i].name ) return &elems[i];
	return NULL;
}


/******************************************************************************
Find a property in the list of properties of a given element.

Entry:
elem      - pointer to element in which we want to find the property
prop_name - name of property to find

Exit:
index - index to position in list
returns a pointer to the property, or NULL if not found
******************************************************************************/

PlyProperty *PlyElement::find_property( const std::string &prop_name , int &index )
{
	for( int i=0 ; i<props.size() ; i++ ) if( prop_name==props[i].prop.name ){ index = i ; return &props[i].prop; }
	index = -1;
	return NULL;
}

/******************************************************************************
Read an element from an ascii file.

Entry:
elem_ptr - pointer to element
******************************************************************************/

void PlyFile::_ascii_get_element( void *elem_ptr )
{
	std::vector< std::string > words;
	PlyElement *elem;
	int which_word;
	void *elem_data , *item=NULL;
	char *item_ptr;
	int item_size;
	int int_val;
	unsigned int uint_val;
	long long longlong_val;
	unsigned long long ulonglong_val;
	double double_val;
	int list_count;
	int store_it;
	char **store_array;
	char *orig_line;
	char *other_data=NULL;
	int other_flag;

	/* the kind of element we're reading currently */
	elem = which_elem;

	/* do we need to setup for other_props? */
	if( elem->other_offset!=NO_OTHER_PROPS )
	{
		char **ptr;
		other_flag = 1;
		/* make room for other_props */
		other_data = (char *)malloc( elem->other_size );
		/* store pointer in user's structure to the other_props */
		ptr = (char **) ( (char*)elem_ptr + elem->other_offset);
		*ptr = other_data;
	}
	else other_flag = 0;

	/* read in the element */
	words = get_words( fp , &orig_line );
	if( !words.size() ) ERROR_OUT( "Unexpected end of file" );

	which_word = 0;

	for( int j=0 ; j<elem->props.size() ; j++ )
	{
		PlyProperty &prop = elem->props[j].prop;
		store_it = (elem->props[j].store | other_flag);

		/* store either in the user's structure or in other_props */
		if( elem->props[j].store ) elem_data = elem_ptr;
		else                       elem_data = other_data;

		if( prop.is_list )       /* a list */
		{
			/* get and store the number of items in the list */
			get_ascii_item( words[which_word++] , prop.count_external , int_val , uint_val , longlong_val , ulonglong_val , double_val );
			if( store_it )
			{
				item = (char *)elem_data + prop.count_offset;
				store_item( item , prop.count_internal , int_val , uint_val , longlong_val , ulonglong_val , double_val );
			}

			/* allocate space for an array of items and store a ptr to the array */
			list_count = int_val;
			item_size = ply_type_size[ prop.internal_type ];
			store_array = (char **)( (char *)elem_data + prop.offset );

			if( list_count==0 )
			{
				if( store_it ) *store_array = NULL;
			}
			else
			{
				if( store_it )
				{
					item_ptr = (char *) malloc (sizeof (char) * item_size * list_count);
					item = item_ptr;
					*store_array = item_ptr;
				}

				/* read items and store them into the array */
				for( int k=0 ; k<list_count ; k++ )
				{
					get_ascii_item( words[which_word++] , prop.external_type , int_val , uint_val , longlong_val , ulonglong_val , double_val );
					if( store_it )
					{
						store_item( item , prop.internal_type , int_val , uint_val , longlong_val , ulonglong_val , double_val );
						item = (char *)item + item_size;
					}
				}
			}
		}
		else                     /* not a list */
		{
			get_ascii_item( words[which_word++] , prop.external_type , int_val , uint_val , longlong_val , ulonglong_val , double_val );
			if( store_it )
			{
				item = (char *)elem_data + prop.offset;
				store_item( item , prop.internal_type , int_val , uint_val , longlong_val , ulonglong_val , double_val );
			}
		}
	}
}

/******************************************************************************
Read an element from a binary file.

Entry:
elem_ptr - pointer to an element
******************************************************************************/

void PlyFile::_binary_get_element( void *elem_ptr )
{
	PlyElement *elem;
	void *elem_data , *item=NULL;
	char *item_ptr;
	int item_size;
	int int_val;
	unsigned int uint_val;
	long long longlong_val;
	unsigned long long ulonglong_val;
	double double_val;
	int list_count;
	int store_it;
	char **store_array;
	char *other_data=NULL;
	int other_flag;

	/* the kind of element we're reading currently */
	elem = which_elem;

	/* do we need to setup for other_props? */
	if( elem->other_offset!=NO_OTHER_PROPS )
	{
		char **ptr;
		other_flag = 1;
		/* make room for other_props */
		other_data = (char *) malloc (elem->other_size);
		/* store pointer in user's structure to the other_props */
		ptr = (char **) ((char *)elem_ptr + elem->other_offset);
		*ptr = other_data;
	}
	else other_flag = 0;

	/* read in a number of elements */

	for( int j=0 ; j<elem->props.size() ; j++ )
	{
		PlyProperty &prop = elem->props[j].prop;
		store_it = ( elem->props[j].store | other_flag );

		/* store either in the user's structure or in other_props */
		if( elem->props[j].store ) elem_data = elem_ptr;
		else                       elem_data = other_data;

		if( prop.is_list )       /* a list */
		{
			/* get and store the number of items in the list */
			get_binary_item( fp , file_type , prop.count_external , int_val, uint_val , longlong_val , ulonglong_val , double_val );
			if( store_it )
			{
				item = (char *)elem_data + prop.count_offset;
				store_item( item , prop.count_internal , int_val , uint_val , longlong_val , ulonglong_val , double_val );
			}

			/* allocate space for an array of items and store a ptr to the array */
			list_count = int_val;
			item_size = ply_type_size[ prop.internal_type ];
			store_array = (char **) ((char *)elem_data + prop.offset);
			if( list_count==0 )
			{
				if( store_it ) *store_array = NULL;
			}
			else
			{
				if( store_it )
				{
					item_ptr = (char *)malloc(sizeof (char) * item_size * list_count);
					item = item_ptr;
					*store_array = item_ptr;
				}

				/* read items and store them into the array */
				for( int k=0 ; k<list_count ; k++ )
				{
					get_binary_item( fp , file_type , prop.external_type , int_val , uint_val , longlong_val , ulonglong_val , double_val );
					if( store_it )
					{
						store_item( item , prop.internal_type , int_val , uint_val , longlong_val , ulonglong_val , double_val );
						item = (char *)item + item_size;
					}
				}
			}
		}
		else                     /* not a list */
		{
			get_binary_item( fp , file_type , prop.external_type , int_val , uint_val , longlong_val , ulonglong_val , double_val );
			if( store_it )
			{
				item = (char *)elem_data + prop.offset;
				store_item( item , prop.internal_type , int_val , uint_val , longlong_val , ulonglong_val , double_val );
			}
		}
	}
}


/******************************************************************************
Write to a file the word that represents a PLY data type.

Entry:
fp   - file pointer
code - code for type
******************************************************************************/

void write_scalar_type( FILE *fp , int code )
{
	/* make sure this is a valid code */
	if( code<=PLY_START_TYPE || code>=PLY_END_TYPE ) ERROR_OUT( "Bad data code: " , code );

	/* write the code to a file */
	fprintf( fp , "%s" , type_names[code] );
}

/******************************************************************************
Reverse the order in an array of bytes.  This is the conversion from big
endian to little endian and vice versa

Entry:
bytes     - array of bytes to reverse (in place)
num_bytes - number of bytes in array
******************************************************************************/

void swap_bytes( void *bytes , int num_bytes )
{
	char *chars = (char *)bytes;

	for( int i=0 ; i<num_bytes/2 ; i++ )
	{
		char temp = chars[i];
		chars[i] = chars[(num_bytes-1)-i];
		chars[(num_bytes-1)-i] = temp;
	}
}

/******************************************************************************
Find out if this machine is big endian or little endian

Exit:
set global variable, native_binary_type =
either PLY_BINARY_BE or PLY_BINARY_LE

******************************************************************************/

void get_native_binary_type( void )
{
	endian_test_type test;

	test.int_value = 0;
	test.int_value = 1;
	if     ( test.byte_values[0]==1 ) native_binary_type = PLY_BINARY_LE;
	else if( test.byte_values[sizeof(int)-1] == 1) native_binary_type = PLY_BINARY_BE;
	else ERROR_OUT( "Couldn't determine machine endianness" );
}

/******************************************************************************
Verify that all the native types are the sizes we need


******************************************************************************/

void check_types()
{
	if( (ply_type_size[PLY_CHAR] != sizeof(char)) ||
		(ply_type_size[PLY_SHORT] != sizeof(short)) ||	
		(ply_type_size[PLY_INT] != sizeof(int)) ||	
		(ply_type_size[PLY_LONGLONG] != sizeof(long long)) ||	
		(ply_type_size[PLY_UCHAR] != sizeof(unsigned char)) ||	
		(ply_type_size[PLY_USHORT] != sizeof(unsigned short)) ||	
		(ply_type_size[PLY_UINT] != sizeof(unsigned int)) ||	
		(ply_type_size[PLY_ULONGLONG] != sizeof(unsigned long long)) ||	
		(ply_type_size[PLY_FLOAT] != sizeof(float)) ||	
		(ply_type_size[PLY_DOUBLE] != sizeof(double)))
		ERROR_OUT( "Type sizes do not match built-in types" );

	types_checked = 1;
}

/******************************************************************************
Get a text line from a file and break it up into words.

IMPORTANT: The calling routine call "free" on the returned pointer once
finished with it.

Entry:
fp - file to read from

Exit:
nwords    - number of words returned
orig_line - the original line of characters
returns a list of words from the line, or NULL if end-of-file
******************************************************************************/

std::vector< std::string > get_words( FILE *fp , char **orig_line )
{
#define BIG_STRING 4096
	static char str[BIG_STRING];
	static char str_copy[BIG_STRING];
	std::vector< std::string > words;
	int max_words = 10;
	int num_words = 0;
	char *ptr , *ptr2;
	char *result;

	/* read in a line */
	result = fgets( str , BIG_STRING , fp );
	if( result==NULL )
	{
		*orig_line = NULL;
		return words;
	}
	/* convert line-feed and tabs into spaces */
	/* (this guarentees that there will be a space before the */
	/*  null character at the end of the string) */

	str[BIG_STRING-2] = ' ';
	str[BIG_STRING-1] = '\0';

	for( ptr=str , ptr2=str_copy ; *ptr!='\0' ; ptr++ , ptr2++ )
	{
		*ptr2 = *ptr;
		// Added line here to manage carriage returns
		if( *ptr == '\t' || *ptr == '\r' )
		{
			*ptr = ' ';
			*ptr2 = ' ';
		}
		else if( *ptr=='\n' )
		{
			*ptr = ' ';
			*ptr2 = '\0';
			break;
		}
	}

	/* find the words in the line */

	ptr = str;
	while( *ptr!='\0' )
	{
		/* jump over leading spaces */
		while( *ptr==' ' ) ptr++;

		/* break if we reach the end */
		if( *ptr=='\0' ) break;

		char *_ptr = ptr;

		/* jump over non-spaces */
		while( *ptr!=' ' ) ptr++;

		/* place a null character here to mark the end of the word */
		*ptr++ = '\0';

		/* save pointer to beginning of word */
		words.push_back( _ptr );
	}

	/* return the list of words */
	*orig_line = str_copy;
	return words;
}

/******************************************************************************
Return the value of an item, given a pointer to it and its type.

Entry:
item - pointer to item
type - data type that "item" points to

Exit:
returns a double-precision float that contains the value of the item
******************************************************************************/

double get_item_value( const void *item , int type )
{
	switch( type )
	{
	case PLY_CHAR:
	case PLY_INT_8:     return (double)*(const               char *)item;
	case PLY_UCHAR:
	case PLY_UINT_8:    return (double)*(const unsigned      char *)item;
	case PLY_SHORT:
	case PLY_INT_16:    return (double)*(const          short int *)item;
	case PLY_USHORT:
	case PLY_UINT_16:   return (double)*(const unsigned short int *)item;
	case PLY_INT:
	case PLY_INT_32:    return (double)*(const                int *)item;
	case PLY_LONGLONG:
	case PLY_INT_64:    return (double)*(const          long long *)item;
	case PLY_UINT:
	case PLY_UINT_32:   return (double)*(const unsigned       int *)item;
	case PLY_ULONGLONG:
	case PLY_UINT_64:   return (double)*(const unsigned long long *)item;
	case PLY_FLOAT:
	case PLY_FLOAT_32:  return (double)*(const              float *)item;
	case PLY_DOUBLE:
	case PLY_FLOAT_64:  return (double)*(const             double *)item;
	default: ERROR_OUT( "Bad type: " , type );
	}
	return 0;
}


/******************************************************************************
Write out an item to a file as raw binary bytes.

Entry:
fp         - file to write to
int_val    - integer version of item
uint_val   - unsigned integer version of item
double_val - double-precision float version of item
type       - data type to write out
******************************************************************************/

void write_binary_item( FILE *fp , int file_type , int int_val , unsigned int uint_val , long long longlong_val , unsigned long long ulonglong_val , double double_val , int type )
{
	unsigned char uchar_val;
	char char_val;
	unsigned short ushort_val;
	short short_val;
	float float_val;
	void *value;

	switch (type) {
	case PLY_CHAR:
	case PLY_INT_8:
		char_val = char(int_val);
		value = &char_val;
		break;
	case PLY_SHORT:
	case PLY_INT_16:
		short_val = short(int_val);
		value = &short_val;
		break;
	case PLY_INT:
	case PLY_INT_32:
		value = &int_val;
		break;
	case PLY_LONGLONG:
	case PLY_INT_64:
		value = &longlong_val;
		break;
	case PLY_UCHAR:
	case PLY_UINT_8:
		uchar_val = (unsigned char)(uint_val);
		value = &uchar_val;
		break;
	case PLY_USHORT:
	case PLY_UINT_16:
		ushort_val = (unsigned short)(uint_val);
		value = &ushort_val;
		break;
	case PLY_UINT:
	case PLY_UINT_32:
		value = &uint_val;
		break;
	case PLY_ULONGLONG:
	case PLY_UINT_64:
		value = &ulonglong_val;
		break;
	case PLY_FLOAT:
	case PLY_FLOAT_32:
		float_val = (float)double_val;
		value = &float_val;
		break;
	case PLY_DOUBLE:
	case PLY_FLOAT_64:
		value = &double_val;
		break;
	default: ERROR_OUT( "Bad type: " , type );
	}


	if( (file_type!=native_binary_type) && (ply_type_size[type]>1) ) swap_bytes( (char *)value , ply_type_size[type] );
	if( fwrite( value , ply_type_size[type] , 1 , fp )!=1 ) ERROR_OUT( "Failed to write binary item" );
}


/******************************************************************************
Write out an item to a file as ascii characters.

Entry:
fp         - file to write to
int_val    - integer version of item
uint_val   - unsigned integer version of item
double_val - double-precision float version of item
type       - data type to write out
******************************************************************************/

void write_ascii_item( FILE *fp , int int_val , unsigned int uint_val , long long longlong_val , unsigned long long ulonglong_val , double double_val , int type )
{
	switch (type)
	{
	case PLY_CHAR:
	case PLY_INT_8:
	case PLY_SHORT:
	case PLY_INT_16:
	case PLY_INT:
	case PLY_INT_32:
		if( fprintf( fp , "%d " , int_val )<=0 ) ERROR_OUT( "fprintf() failed -- aborting" );
		break;
	case PLY_LONGLONG:
	case PLY_INT_64:
		if( fprintf( fp , "%lld " , longlong_val )<=0 ) ERROR_OUT( "fprintf() failed -- aborting" );
		break;
	case PLY_UCHAR:
	case PLY_UINT_8:
	case PLY_USHORT:
	case PLY_UINT_16:
	case PLY_UINT:
	case PLY_UINT_32:
		if( fprintf( fp , "%u " , uint_val )<=0 ) ERROR_OUT( "fprintf() failed -- aborting" );
		break;
	case PLY_ULONGLONG:
	case PLY_UINT_64:
		if( fprintf( fp , "%llu " , ulonglong_val )<=0 ) ERROR_OUT( "fprintf() failed -- aborting" );
		break;
	case PLY_FLOAT:
	case PLY_FLOAT_32:
	case PLY_DOUBLE:
	case PLY_FLOAT_64:
		if( fprintf( fp , "%g " , double_val )<=0 ) ERROR_OUT( "fprintf() failed -- aborting" );
		break;
	default: ERROR_OUT( "Bad type: " , type );
	}
}

/******************************************************************************
Get the value of an item that is in memory, and place the result
into an integer, an unsigned integer and a double.

Entry:
ptr  - pointer to the item
type - data type supposedly in the item

Exit:
int_val    - integer value
uint_val   - unsigned integer value
double_val - double-precision floating point value
******************************************************************************/

void get_stored_item( void *ptr , int type , int &int_val , unsigned int &uint_val , long long &longlong_val , unsigned long long &ulonglong_val , double &double_val )
{
	switch( type )
	{
	case PLY_CHAR:
	case PLY_INT_8:
		int_val = *((char *) ptr);
		uint_val = int_val;
		double_val = int_val;
		longlong_val = (long long)int_val;
		ulonglong_val = (unsigned long long)int_val;
		break;
	case PLY_UCHAR:
	case PLY_UINT_8:
		uint_val = *((unsigned char *) ptr);
		int_val = uint_val;
		double_val = uint_val;
		longlong_val = (long long)uint_val;
		ulonglong_val = (unsigned long long)uint_val;
		break;
	case PLY_SHORT:
	case PLY_INT_16:
		int_val = *((short int *) ptr);
		uint_val = int_val;
		double_val = int_val;
		longlong_val = (long long)int_val;
		ulonglong_val = (unsigned long long)int_val;
		break;
	case PLY_USHORT:
	case PLY_UINT_16:
		uint_val = *((unsigned short int *) ptr);
		int_val = uint_val;
		double_val = uint_val;
		longlong_val = (long long)uint_val;
		ulonglong_val = (unsigned long long)uint_val;
		break;
	case PLY_INT:
	case PLY_INT_32:
		int_val = *((int *) ptr);
		uint_val = int_val;
		double_val = int_val;
		longlong_val = (long long)int_val;
		ulonglong_val = (unsigned long long)int_val;
		break;
	case PLY_UINT:
	case PLY_UINT_32:
		uint_val = *((unsigned int *) ptr);
		int_val = uint_val;
		double_val = uint_val;
		longlong_val = (long long)uint_val;
		ulonglong_val = (unsigned long long)uint_val;
		break;
	case PLY_LONGLONG:
	case PLY_INT_64:
		longlong_val = *((long long *) ptr);
		ulonglong_val = (unsigned long long)longlong_val;
		int_val = (int)longlong_val;
		uint_val = (unsigned int)longlong_val;
		double_val = (double)longlong_val;
		break;
	case PLY_ULONGLONG:
	case PLY_UINT_64:
		ulonglong_val = *((unsigned long long *) ptr);
		longlong_val = (long long)ulonglong_val;
		int_val = (int)ulonglong_val;
		uint_val = (unsigned int)ulonglong_val;
		double_val = (double)ulonglong_val;
		break;
	case PLY_FLOAT:
	case PLY_FLOAT_32:
		double_val = *((float *) ptr);
		int_val = (int)double_val;
		uint_val = (unsigned int)double_val;
		longlong_val = (long long)double_val;
		ulonglong_val = (unsigned long long)double_val;
		break;
	case PLY_DOUBLE:
	case PLY_FLOAT_64:
		double_val = *((double *) ptr);
		int_val = (int)double_val;
		uint_val = (unsigned int)double_val;
		longlong_val = (long long)double_val;
		ulonglong_val = (unsigned long long)double_val;
		break;
	default: ERROR_OUT( "Bad type: " , type );
	}
}

/******************************************************************************
Get the value of an item from a binary file, and place the result
into an integer, an unsigned integer and a double.

Entry:
fp   - file to get item from
type - data type supposedly in the word

Exit:
int_val    - integer value
uint_val   - unsigned integer value
double_val - double-precision floating point value
******************************************************************************/

void get_binary_item( FILE *fp , int file_type , int type , int &int_val , unsigned int &uint_val , long long &longlong_val , unsigned long long &ulonglong_val , double &double_val )
{
	char c[8];
	void *ptr;

	ptr = ( void * )c;

	if( fread( ptr , ply_type_size[type] , 1 , fp )!=1 ) ERROR_OUT( "fread() failed -- aborting." );
	if( ( file_type!=native_binary_type ) && ( ply_type_size[type]>1 ) ) swap_bytes( (char *)ptr , ply_type_size[type] );

	switch( type )
	{
	case PLY_CHAR:
	case PLY_INT_8:
		int_val = *((char *) ptr);
		uint_val = int_val;
		longlong_val = int_val;
		ulonglong_val = int_val;
		double_val = int_val;
		break;
	case PLY_UCHAR:
	case PLY_UINT_8:
		uint_val = *((unsigned char *) ptr);
		int_val = uint_val;
		longlong_val = int_val;
		ulonglong_val = int_val;
		double_val = uint_val;
		break;
	case PLY_SHORT:
	case PLY_INT_16:
		int_val = *((short int *) ptr);
		uint_val = int_val;
		longlong_val = int_val;
		ulonglong_val = int_val;
		double_val = int_val;
		break;
	case PLY_USHORT:
	case PLY_UINT_16:
		uint_val = *((unsigned short int *) ptr);
		int_val = uint_val;
		longlong_val = int_val;
		ulonglong_val = int_val;
		double_val = uint_val;
		break;
	case PLY_INT:
	case PLY_INT_32:
		int_val = *((int *) ptr);
		uint_val = int_val;
		longlong_val = int_val;
		ulonglong_val = int_val;
		double_val = int_val;
		break;
	case PLY_UINT:
	case PLY_UINT_32:
		uint_val = *((unsigned int *) ptr);
		int_val = uint_val;
		longlong_val = int_val;
		ulonglong_val = int_val;
		double_val = uint_val;
		break;
	case PLY_LONGLONG:
	case PLY_INT_64:
		longlong_val = *((long long *) ptr);
		ulonglong_val = (unsigned long long)longlong_val;
		int_val = (int)longlong_val;
		uint_val = (unsigned int)longlong_val;
		double_val = (double)longlong_val;
		break;
	case PLY_ULONGLONG:
	case PLY_UINT_64:
		ulonglong_val = *((unsigned long long *) ptr);
		longlong_val = (long long)ulonglong_val;
		int_val = (int)ulonglong_val;
		uint_val = (unsigned int)ulonglong_val;
		double_val = (double)ulonglong_val;
		break;
	case PLY_FLOAT:
	case PLY_FLOAT_32:
		double_val = *((float *) ptr);
		int_val = (int)double_val;
		uint_val = (unsigned int)double_val;
		longlong_val = (long long)double_val;
		ulonglong_val = (unsigned long long)int_val;
		break;
	case PLY_DOUBLE:
	case PLY_FLOAT_64:
		double_val = *((double *) ptr);
		int_val = (int)double_val;
		uint_val = (unsigned int)double_val;
		longlong_val = (long long)double_val;
		ulonglong_val = (unsigned long long)int_val;
		break;
	default: ERROR_OUT( "Bad type: " , type );
	}
}

/******************************************************************************
Extract the value of an item from an ascii word, and place the result
into an integer, an unsigned integer and a double.

Entry:
word - word to extract value from
type - data type supposedly in the word

Exit:
int_val    - integer value
uint_val   - unsigned integer value
double_val - double-precision floating point value
******************************************************************************/
void get_ascii_item( const std::string &word , int type , int &int_val , unsigned int &uint_val , long long &longlong_val , unsigned long long &ulonglong_val , double &double_val )
{
	switch( type )
	{
	case PLY_CHAR:
	case PLY_INT_8:
	case PLY_UCHAR:
	case PLY_UINT_8:
	case PLY_SHORT:
	case PLY_INT_16:
	case PLY_USHORT:
	case PLY_UINT_16:
	case PLY_INT:
	case PLY_INT_32:
		int_val = atoi( word.c_str() );
		uint_val = (unsigned int)int_val;
		double_val = (double)int_val;
		longlong_val = (long long)int_val;
		ulonglong_val = (unsigned long long)int_val;
		break;

	case PLY_UINT:
	case PLY_UINT_32:
		uint_val = strtol( word.c_str() , (char **)NULL , 10 );
		int_val = (int)uint_val;
		double_val = (double)uint_val;
		longlong_val = (long long)uint_val;
		ulonglong_val = (unsigned long long)uint_val;
		break;
	case PLY_LONGLONG:
	case PLY_INT_64:
		longlong_val = std::stoll( word.c_str() );
		ulonglong_val = (unsigned long long)longlong_val;
		int_val = (int)longlong_val;
		uint_val = (unsigned int)longlong_val;
		double_val = (double)longlong_val;
		break;
	case PLY_ULONGLONG:
	case PLY_UINT_64:
		ulonglong_val = std::stoull( word.c_str() );
		longlong_val = (long long)ulonglong_val;
		int_val = (int)ulonglong_val;
		uint_val = (unsigned int)ulonglong_val;
		double_val = (double)ulonglong_val;
		break;
	case PLY_FLOAT:
	case PLY_FLOAT_32:
	case PLY_DOUBLE:
	case PLY_FLOAT_64:
		double_val = atof( word.c_str() );
		int_val = (int)double_val;
		uint_val = (unsigned int)double_val;
		longlong_val = (long long)double_val;
		ulonglong_val = (unsigned long long)double_val;
		break;
	default: ERROR_OUT( "Bad type: " , type );
	}
}

/******************************************************************************
Store a value into a place being pointed to, guided by a data type.

Entry:
item       - place to store value
type       - data type
int_val    - integer version of value
uint_val   - unsigned integer version of value
double_val - double version of value

Exit:
item - pointer to stored value
******************************************************************************/

void store_item( void *item , int type , int int_val , unsigned int uint_val , long long longlong_val , unsigned long long ulonglong_val , double double_val )
{
	switch( type )
	{
	case PLY_CHAR:
	case PLY_INT_8:   *(          char *)item = (          char)   int_val ; break;
	case PLY_UCHAR:
	case PLY_UINT_8:  *(unsigned  char *)item = (unsigned  char)  uint_val ; break;
	case PLY_SHORT:
	case PLY_INT_16:  *(         short *)item = (         short)   int_val ; break;
	case PLY_USHORT:
	case PLY_UINT_16: *(unsigned short *)item = (unsigned short)  uint_val ; break;
	case PLY_INT:
	case PLY_INT_32:  *(           int *)item = (           int)   int_val ; break;
	case PLY_UINT:
	case PLY_UINT_32: *(unsigned   int *)item = (unsigned   int)  uint_val ; break;
	case PLY_LONGLONG:
	case PLY_INT_64:  *(     long long *)item = (     long long)longlong_val ; break;
	case PLY_ULONGLONG:
	case PLY_UINT_64: *(unsigned long long *)item = (unsigned long long)ulonglong_val ; break;
	case PLY_FLOAT:
	case PLY_FLOAT_32: *(        float *)item = (         float)double_val ; break;
	case PLY_DOUBLE:
	case PLY_FLOAT_64: *(       double *)item = (        double)double_val ; break;
	default: ERROR_OUT( "Bad type: " , type );
	}
}


/******************************************************************************
Add an element to a PLY file descriptor.

Entry:
plyfile - PLY file descriptor
words   - list of words describing the element
nwords  - number of words in the list
******************************************************************************/

void PlyFile::add_element( const std::vector< std::string > &words )
{
	PlyElement elem;

	/* set the new element */
	elem.name = words[1];
	elem.num = std::stoull( words[2] );
	elem.props.resize(0);

	/* add the new element to the object's list */
	elems.push_back( elem );
}

/******************************************************************************
Return the type of a property, given the name of the property.

Entry:
name - name of property type

Exit:
returns integer code for property, or 0 if not found
******************************************************************************/

int get_prop_type( const std::string &type_name )
{
	for( int i=PLY_START_TYPE+1 ; i<PLY_END_TYPE ; i++ ) if( type_name==type_names[i] ) return i;

	/* if we get here, we didn't find the type */
	return 0;
}

/******************************************************************************
Add a property to a PLY file descriptor.

Entry:
plyfile - PLY file descriptor
words   - list of words describing the property
nwords  - number of words in the list
******************************************************************************/

void PlyFile::add_property( const std::vector< std::string > &words )
{
	PlyProperty prop;
	if( words[1]=="list" )       /* is a list */
	{
		prop.count_external = get_prop_type( words[2] );
		prop.external_type = get_prop_type( words[3]) ;
		prop.name = words[4];
		prop.is_list = 1;
	}
	else         /* not a list */
	{
		prop.external_type = get_prop_type( words[1] );
		prop.name = words[2];
		prop.is_list = 0;
	}

	/* add this property to the list of properties of the current element */
	elems.back().props.push_back( PlyStoredProperty( prop , DONT_STORE_PROP ) );
}


/******************************************************************************
Add a comment to a PLY file descriptor.

Entry:
plyfile - PLY file descriptor
line    - line containing comment
******************************************************************************/

void PlyFile::add_comment( const std::string &line )
{
	/* skip over "comment" and leading spaces and tabs */
	int i = 7;
	while( line[i]==' ' || line[i] =='\t' ) i++;

	put_comment( line.substr(i) );
}


/******************************************************************************
Add a some object information to a PLY file descriptor.

Entry:
plyfile - PLY file descriptor
line    - line containing text info
******************************************************************************/

void PlyFile::add_obj_info( const std::string &line )
{
	/* skip over "obj_info" and leading spaces and tabs */
	int i = 8;
	while( line[i]==' ' || line[i]=='\t' ) i++;
	put_obj_info( line.substr(i) );
}
