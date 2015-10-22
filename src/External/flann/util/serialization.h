#ifndef SERIALIZATION_H_
#define SERIALIZATION_H_

#include <vector>
#include <map>
#include <stdio.h>

namespace flann
{
namespace serialization
{

struct access 
{
    template<typename Archive, typename T>
    static inline void serialize(Archive& ar, T& type)
    {
        type.serialize(ar);
    }
};


template<typename Archive, typename T>
inline void serialize(Archive& ar, T& type)
{
    access::serialize(ar,type);
}

template<typename T>
struct Serializer
{
    template<typename InputArchive>
    static inline void load(InputArchive& ar, T& val)
    {
        serialization::serialize(ar,val);
    }
    template<typename OutputArchive>
    static inline void save(OutputArchive& ar, const T& val)
    {
        serialization::serialize(ar,const_cast<T&>(val));
    }
};

#define BASIC_TYPE_SERIALIZER(type)\
template<> \
struct Serializer<type> \
{\
    template<typename InputArchive>\
    static inline void load(InputArchive& ar, type& val)\
    {\
        ar.load(val);\
    }\
    template<typename OutputArchive>\
    static inline void save(OutputArchive& ar, const type& val)\
    {\
        ar.save(val);\
    }\
}

#define ENUM_SERIALIZER(type)\
template<>\
struct Serializer<type>\
{\
    template<typename InputArchive>\
    static inline void load(InputArchive& ar, type& val)\
    {\
        int int_val;\
        ar & int_val;\
        val = (type) int_val;\
    }\
    template<typename OutputArchive>\
    static inline void save(OutputArchive& ar, const type& val)\
    {\
        int int_val = (int)val;\
        ar & int_val;\
    }\
}


// declare serializers for simple types
BASIC_TYPE_SERIALIZER(char);
BASIC_TYPE_SERIALIZER(unsigned char);
BASIC_TYPE_SERIALIZER(short);
BASIC_TYPE_SERIALIZER(unsigned short);
BASIC_TYPE_SERIALIZER(int);
BASIC_TYPE_SERIALIZER(unsigned int);
BASIC_TYPE_SERIALIZER(long);
BASIC_TYPE_SERIALIZER(unsigned long);
BASIC_TYPE_SERIALIZER(float);
BASIC_TYPE_SERIALIZER(double);
BASIC_TYPE_SERIALIZER(bool);



// serializer for std::vector
template<typename T>
struct Serializer<std::vector<T> >
{
    template<typename InputArchive>
    static inline void load(InputArchive& ar, std::vector<T>& val)
    {
        size_t size;
        ar & size;
        val.resize(size);
        for (size_t i=0;i<size;++i) {
            ar & val[i];
        }
    }

    template<typename OutputArchive>
    static inline void save(OutputArchive& ar, const std::vector<T>& val)
    {
        ar & val.size();
        for (size_t i=0;i<val.size();++i) {
            ar & val[i];
        }
    }
};

// serializer for std::vector
template<typename K, typename V>
struct Serializer<std::map<K,V> >
{
    template<typename InputArchive>
    static inline void load(InputArchive& ar, std::map<K,V>& map_val)
    {
        size_t size;
        ar & size;
        for (size_t i = 0; i < size; ++i)
        {
            K key;
            ar & key;
            V value;
            ar & value;
            map_val[key] = value;
        }
    }

    template<typename OutputArchive>
    static inline void save(OutputArchive& ar, const std::map<K,V>& map_val)
    {
        ar & map_val.size();
        for (typename std::map<K,V>::const_iterator i=map_val.begin(); i!=map_val.end(); ++i) {
            ar & i->first;
            ar & i->second;
        }
    }
};

template<typename T>
struct Serializer<T*>
{
    template<typename InputArchive>
    static inline void load(InputArchive& ar, T*& val)
    {
        ar.load(val);
    }

    template<typename OutputArchive>
    static inline void save(OutputArchive& ar,  T* const& val)
    {
        ar.save(val);
    }
};

template<typename T, int N>
struct Serializer<T[N]>
{
    template<typename InputArchive>
    static inline void load(InputArchive& ar, T (&val)[N])
    {
        ar.load(val);
    }

    template<typename OutputArchive>
    static inline void save(OutputArchive& ar,  T const (&val)[N])
    {
        ar.save(val);
    }
};




struct binary_object
{
    void const * ptr_;
    size_t size_;

    binary_object( void * const ptr, size_t size) :
        ptr_(ptr),
        size_(size)
    {}
    binary_object(const binary_object & rhs) :
        ptr_(rhs.ptr_),
        size_(rhs.size_)
    {}

    binary_object & operator=(const binary_object & rhs) {
        ptr_ = rhs.ptr_;
        size_ = rhs.size_;
        return *this;
    }
};

inline const binary_object make_binary_object(/* const */ void * t, size_t size){
    return binary_object(t, size);
}

template<>
struct Serializer<const binary_object>
{
    template<typename InputArchive>
    static inline void load(InputArchive& ar, const binary_object& b)
    {
        ar.load_binary(const_cast<void *>(b.ptr_), b.size_);
    }

    template<typename OutputArchive>
    static inline void save(OutputArchive& ar,  const binary_object& b)
    {
        ar.save_binary(b.ptr_, b.size_);
    }
};

template<>
struct Serializer<binary_object>
{
    template<typename InputArchive>
    static inline void load(InputArchive& ar, binary_object& b)
    {
        ar.load_binary(const_cast<void *>(b.ptr_), b.size_);
    }

    template<typename OutputArchive>
    static inline void save(OutputArchive& ar,  const binary_object& b)
    {
        ar.save_binary(b.ptr_, b.size_);
    }
};



template <bool C_> 
struct bool_ {
    static const bool value = C_;
    typedef bool value_type;
};


class ArchiveBase
{
public:

	void* getObject() { return object_; }

	void setObject(void* object) { object_ = object; }

private:
	void* object_;
};


template<typename Archive>
class InputArchive : public ArchiveBase
{
protected:
    InputArchive() {};
public:
    typedef bool_<true> is_loading;
    typedef bool_<false> is_saving;

    template<typename T>
    Archive& operator& (T& val)
    {
        Serializer<T>::load(*static_cast<Archive*>(this),val);
        return *static_cast<Archive*>(this);
    }
};


template<typename Archive>
class OutputArchive : public ArchiveBase
{
protected:
    OutputArchive() {};
public:
    typedef bool_<false> is_loading;
    typedef bool_<true> is_saving;
    
    template<typename T>
    Archive& operator& (const T& val)
    {
        Serializer<T>::save(*static_cast<Archive*>(this),val);
        return *static_cast<Archive*>(this);
    }
};



class SizeArchive : public OutputArchive<SizeArchive>
{
    size_t size_;
public:

    SizeArchive() : size_(0)
    {
    }

    template<typename T>
    void save(const T& val)
    {
        size_ += sizeof(val);
    }

    template<typename T>
    void save_binary(T* ptr, size_t size)
    {
    	size_ += size;
    }


    void reset()
    {
        size_ = 0;
    }

    size_t size()
    {
        return size_;
    }
};


//
//class PrintArchive : public OutputArchive<PrintArchive>
//{
//public:
//    template<typename T>
//    void save(const T& val)
//    {
//        std::cout << val << std::endl;
//    }
//
//    template<typename T>
//    void save_binary(T* ptr, size_t size)
//    {
//        std::cout << "<binary object>" << std::endl;
//    }
//};


class SaveArchive : public OutputArchive<SaveArchive>
{
    FILE* stream_;
    bool own_stream_;
public:
    SaveArchive(const char* filename)
    {
        stream_ = fopen(filename, "w");
        own_stream_ = true;
    }

    SaveArchive(FILE* stream) : stream_(stream), own_stream_(false)
    {
    }

    ~SaveArchive()
    {
    	if (own_stream_) {
    		fclose(stream_);
    	}
    }

    template<typename T>
    void save(const T& val)
    {
        fwrite(&val, sizeof(val), 1, stream_);
    }

    template<typename T>
    void save(T* const& val)
    {
    	// don't save pointers
        //fwrite(&val, sizeof(val), 1, handle_);
    }
    
    template<typename T>
    void save_binary(T* ptr, size_t size)
    {
    	fwrite(ptr, size, 1, stream_);
    }

};


class LoadArchive : public InputArchive<LoadArchive>
{
    FILE* stream_;
    bool own_stream_;
public:
    LoadArchive(const char* filename)
    {
        stream_ = fopen(filename, "r");
        own_stream_ = true;
    }

    LoadArchive(FILE* stream) : stream_(stream), own_stream_(false)
    {
    }

    ~LoadArchive()
    {
    	if (own_stream_) {
    		fclose(stream_);
    	}
    }

    template<typename T>
    void load(T& val)
    {
        size_t ret = fread(&val, sizeof(val), 1, stream_);
        if (ret!=1) {
        	throw FLANNException("Error loading from file");
        }
    }

    template<typename T>
    void load(T*& val)
    {
    	// don't load pointers
        //fread(&val, sizeof(val), 1, handle_);
    }

    template<typename T>
    void load_binary(T* ptr, size_t size)
    {
    	size_t ret = fread(ptr, size, 1, stream_);
        if (ret!=1) {
        	throw FLANNException("Error loading from file");
        }
    }


};

} // namespace serialization
} // namespace flann
#endif // SERIALIZATION_H_
