/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/


#pragma once

#if defined(_WIN32) || defined(_WIN64)
    #include <windows.h>
    #undef small            // Windows is terrible for polluting macro namespace
#else
    #include <sys/resource.h>
#endif

#include <cuda_runtime.h>

#include <stdio.h>
#include <math.h>
#include <float.h>

#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <limits>

#include "mersenne.h"
#include "half.h"

#include "cub/util_debug.cuh"
#include "cub/util_device.cuh"
#include "cub/util_type.cuh"
#include "cub/util_macro.cuh"
#include "cub/iterator/discard_output_iterator.cuh"


/******************************************************************************
 * Assertion macros
 ******************************************************************************/

/**
 * Assert equals
 */
#define AssertEquals(a, b) if ((a) != (b)) { std::cerr << "\n(" << __FILE__ << ": " << __LINE__ << ")\n"; exit(1);}


/******************************************************************************
 * Command-line parsing functionality
 ******************************************************************************/

/**
 * Utility for parsing command line arguments
 */
struct CommandLineArgs
{

    std::vector<std::string>    keys;
    std::vector<std::string>    values;
    std::vector<std::string>    args;
    cudaDeviceProp              deviceProp;
    float                       device_giga_bandwidth;
    size_t                      device_free_physmem;
    size_t                      device_total_physmem;

    /**
     * Constructor
     */
    CommandLineArgs(int argc, char **argv) :
        keys(10),
        values(10)
    {
        using namespace std;

        // Initialize mersenne generator
        unsigned int mersenne_init[4]=  {0x123, 0x234, 0x345, 0x456};
        mersenne::init_by_array(mersenne_init, 4);

        for (int i = 1; i < argc; i++)
        {
            string arg = argv[i];

            if ((arg[0] != '-') || (arg[1] != '-'))
            {
                args.push_back(arg);
                continue;
            }

            string::size_type pos;
            string key, val;
            if ((pos = arg.find('=')) == string::npos) {
                key = string(arg, 2, arg.length() - 2);
                val = "";
            } else {
                key = string(arg, 2, pos - 2);
                val = string(arg, pos + 1, arg.length() - 1);
            }

            keys.push_back(key);
            values.push_back(val);
        }
    }


    /**
     * Checks whether a flag "--<flag>" is present in the commandline
     */
    bool CheckCmdLineFlag(const char* arg_name)
    {
        using namespace std;

        for (int i = 0; i < int(keys.size()); ++i)
        {
            if (keys[i] == string(arg_name))
                return true;
        }
        return false;
    }


    /**
     * Returns number of naked (non-flag and non-key-value) commandline parameters
     */
    template <typename T>
    int NumNakedArgs()
    {
        return args.size();
    }


    /**
     * Returns the commandline parameter for a given index (not including flags)
     */
    template <typename T>
    void GetCmdLineArgument(int index, T &val)
    {
        using namespace std;
        if (index < args.size()) {
            istringstream str_stream(args[index]);
            str_stream >> val;
        }
    }

    /**
     * Returns the value specified for a given commandline parameter --<flag>=<value>
     */
    template <typename T>
    void GetCmdLineArgument(const char *arg_name, T &val)
    {
        using namespace std;

        for (int i = 0; i < int(keys.size()); ++i)
        {
            if (keys[i] == string(arg_name))
            {
                istringstream str_stream(values[i]);
                str_stream >> val;
            }
        }
    }


    /**
     * Returns the values specified for a given commandline parameter --<flag>=<value>,<value>*
     */
    template <typename T>
    void GetCmdLineArguments(const char *arg_name, std::vector<T> &vals)
    {
        using namespace std;

        if (CheckCmdLineFlag(arg_name))
        {
            // Clear any default values
            vals.clear();

            // Recover from multi-value string
            for (int i = 0; i < keys.size(); ++i)
            {
                if (keys[i] == string(arg_name))
                {
                    string val_string(values[i]);
                    istringstream str_stream(val_string);
                    string::size_type old_pos = 0;
                    string::size_type new_pos = 0;

                    // Iterate comma-separated values
                    T val;
                    while ((new_pos = val_string.find(',', old_pos)) != string::npos)
                    {
                        if (new_pos != old_pos)
                        {
                            str_stream.width(new_pos - old_pos);
                            str_stream >> val;
                            vals.push_back(val);
                        }

                        // skip over comma
                        str_stream.ignore(1);
                        old_pos = new_pos + 1;
                    }

                    // Read last value
                    str_stream >> val;
                    vals.push_back(val);
                }
            }
        }
    }


    /**
     * The number of pairs parsed
     */
    int ParsedArgc()
    {
        return (int) keys.size();
    }

    /**
     * Initialize device
     */
    cudaError_t DeviceInit(int dev = -1)
    {
        cudaError_t error = cudaSuccess;

        do
        {
            int deviceCount;
            error = CubDebug(cudaGetDeviceCount(&deviceCount));
            if (error) break;

            if (deviceCount == 0) {
                fprintf(stderr, "No devices supporting CUDA.\n");
                exit(1);
            }
            if (dev < 0)
            {
                GetCmdLineArgument("device", dev);
            }
            if ((dev > deviceCount - 1) || (dev < 0))
            {
                dev = 0;
            }

            error = CubDebug(cudaSetDevice(dev));
            if (error) break;

            CubDebugExit(cudaMemGetInfo(&device_free_physmem, &device_total_physmem));

            int ptx_version;
            error = CubDebug(cub::PtxVersion(ptx_version));
            if (error) break;

            error = CubDebug(cudaGetDeviceProperties(&deviceProp, dev));
            if (error) break;

            if (deviceProp.major < 1) {
                fprintf(stderr, "Device does not support CUDA.\n");
                exit(1);
            }

            device_giga_bandwidth = float(deviceProp.memoryBusWidth) * deviceProp.memoryClockRate * 2 / 8 / 1000 / 1000;

            if (!CheckCmdLineFlag("quiet"))
            {
                printf(
                        "Using device %d: %s (PTX version %d, SM%d, %d SMs, "
                        "%lld free / %lld total MB physmem, "
                        "%.3f GB/s @ %d kHz mem clock, ECC %s)\n",
                    dev,
                    deviceProp.name,
                    ptx_version,
                    deviceProp.major * 100 + deviceProp.minor * 10,
                    deviceProp.multiProcessorCount,
                    (unsigned long long) device_free_physmem / 1024 / 1024,
                    (unsigned long long) device_total_physmem / 1024 / 1024,
                    device_giga_bandwidth,
                    deviceProp.memoryClockRate,
                    (deviceProp.ECCEnabled) ? "on" : "off");
                fflush(stdout);
            }

        } while (0);

        return error;
    }
};

/******************************************************************************
 * Random bits generator
 ******************************************************************************/

int g_num_rand_samples = 0;


template <typename T>
bool IsNaN(T val) { return false; }

template<>
__noinline__ bool IsNaN<float>(float val)
{
    volatile unsigned int bits = reinterpret_cast<unsigned int &>(val);

    return (((bits >= 0x7F800001) && (bits <= 0x7FFFFFFF)) || 
        ((bits >= 0xFF800001) && (bits <= 0xFFFFFFFF)));
}

template<>
__noinline__ bool IsNaN<float1>(float1 val)
{
    return (IsNaN(val.x));
}

template<>
__noinline__ bool IsNaN<float2>(float2 val)
{
    return (IsNaN(val.y) || IsNaN(val.x));
}

template<>
__noinline__ bool IsNaN<float3>(float3 val)
{
    return (IsNaN(val.z) || IsNaN(val.y) || IsNaN(val.x));
}

template<>
__noinline__ bool IsNaN<float4>(float4 val)
{
    return (IsNaN(val.y) || IsNaN(val.x) || IsNaN(val.w) || IsNaN(val.z));
}

template<>
__noinline__ bool IsNaN<double>(double val)
{
    volatile unsigned long long bits = *reinterpret_cast<unsigned long long *>(&val);

    return (((bits >= 0x7FF0000000000001) && (bits <= 0x7FFFFFFFFFFFFFFF)) || 
        ((bits >= 0xFFF0000000000001) && (bits <= 0xFFFFFFFFFFFFFFFF)));
}

template<>
__noinline__ bool IsNaN<double1>(double1 val)
{
    return (IsNaN(val.x));
}

template<>
__noinline__ bool IsNaN<double2>(double2 val)
{
    return (IsNaN(val.y) || IsNaN(val.x));
}

template<>
__noinline__ bool IsNaN<double3>(double3 val)
{
    return (IsNaN(val.z) || IsNaN(val.y) || IsNaN(val.x));
}

template<>
__noinline__ bool IsNaN<double4>(double4 val)
{
    return (IsNaN(val.y) || IsNaN(val.x) || IsNaN(val.w) || IsNaN(val.z));
}


template<>
__noinline__ bool IsNaN<half_t>(half_t val)
{
    volatile unsigned short bits = reinterpret_cast<unsigned short &>(val);

    return (((bits >= 0x7C01) && (bits <= 0x7FFF)) ||
        ((bits >= 0xFC01) && (bits <= 0xFFFFFFFF)));
}



/**
 * Generates random keys.
 *
 * We always take the second-order byte from rand() because the higher-order
 * bits returned by rand() are commonly considered more uniformly distributed
 * than the lower-order bits.
 *
 * We can decrease the entropy level of keys by adopting the technique
 * of Thearling and Smith in which keys are computed from the bitwise AND of
 * multiple random samples:
 *
 * entropy_reduction    | Effectively-unique bits per key
 * -----------------------------------------------------
 * -1                   | 0
 * 0                    | 32
 * 1                    | 25.95 (81%)
 * 2                    | 17.41 (54%)
 * 3                    | 10.78 (34%)
 * 4                    | 6.42 (20%)
 * ...                  | ...
 *
 */
template <typename K>
void RandomBits(
    K &key,
    int entropy_reduction = 0,
    int begin_bit = 0,
    int end_bit = sizeof(K) * 8)
{
    const int NUM_BYTES = sizeof(K);
    const int WORD_BYTES = sizeof(unsigned int);
    const int NUM_WORDS = (NUM_BYTES + WORD_BYTES - 1) / WORD_BYTES;

    unsigned int word_buff[NUM_WORDS];

    if (entropy_reduction == -1)
    {
        memset((void *) &key, 0, sizeof(key));
        return;
    }

    if (end_bit < 0)
        end_bit = sizeof(K) * 8;

    while (true) 
    {
        // Generate random word_buff
        for (int j = 0; j < NUM_WORDS; j++)
        {
            int current_bit = j * WORD_BYTES * 8;

            unsigned int word = 0xffffffff;
            word &= 0xffffffff << CUB_MAX(0, begin_bit - current_bit);
            word &= 0xffffffff >> CUB_MAX(0, (current_bit + (WORD_BYTES * 8)) - end_bit);

            for (int i = 0; i <= entropy_reduction; i++)
            {
                // Grab some of the higher bits from rand (better entropy, supposedly)
                word &= mersenne::genrand_int32();
                g_num_rand_samples++;                
            }

            word_buff[j] = word;
        }

        memcpy(&key, word_buff, sizeof(K));

        K copy = key;
        if (!IsNaN(copy))
            break;          // avoids NaNs when generating random floating point numbers
    }
}

/// Randomly select number between [0:max)
template <typename T>
T RandomValue(T max)
{
    unsigned int bits;
    unsigned int max_int = (unsigned int) -1;
    do {
        RandomBits(bits);
    } while (bits == max_int);

    return (T) ((double(bits) / double(max_int)) * double(max));
}


/******************************************************************************
 * Console printing utilities
 ******************************************************************************/

/**
 * Helper for casting character types to integers for cout printing
 */
template <typename T>
T CoutCast(T val) { return val; }

int CoutCast(char val) { return val; }

int CoutCast(unsigned char val) { return val; }

int CoutCast(signed char val) { return val; }



/******************************************************************************
 * Test value initialization utilities
 ******************************************************************************/

/**
 * Test problem generation options
 */
enum GenMode
{
    UNIFORM,            // Assign to '2', regardless of integer seed
    INTEGER_SEED,       // Assign to integer seed
    RANDOM,             // Assign to random, regardless of integer seed
};

/**
 * Initialize value
 */
template <typename T>
__host__ __device__ __forceinline__ void InitValue(GenMode gen_mode, T &value, int index = 0)
{
    switch (gen_mode)
    {
#if (CUB_PTX_ARCH == 0)
    case RANDOM:
         RandomBits(value);
         break;
#endif
     case UNIFORM:
        value = 2;
        break;
    case INTEGER_SEED:
    default:
         value = (T) index;
        break;
    }
}


/**
 * Initialize value (bool)
 */
__host__ __device__ __forceinline__ void InitValue(GenMode gen_mode, bool &value, int index = 0)
{
    switch (gen_mode)
    {
#if (CUB_PTX_ARCH == 0)
    case RANDOM:
        char c;
        RandomBits(c, 0, 0, 1);
        value = (c > 0);
        break;
#endif
     case UNIFORM:
        value = true;
        break;
    case INTEGER_SEED:
    default:
        value = (index > 0);
        break;
    }
}


/**
 * cub::NullType test initialization
 */
__host__ __device__ __forceinline__ void InitValue(GenMode gen_mode, cub::NullType &value, int index = 0)
{}


/**
 * cub::KeyValuePair<OffsetT, ValueT>test initialization
 */
template <typename KeyT, typename ValueT>
__host__ __device__ __forceinline__ void InitValue(
    GenMode                             gen_mode,
    cub::KeyValuePair<KeyT, ValueT>&    value,
    int                                 index = 0)
{
    InitValue(gen_mode, value.value, index);

    // Assign corresponding flag with a likelihood of the last bit being set with entropy-reduction level 3
    RandomBits(value.key, 3);
    value.key = (value.key & 0x1);
}



/******************************************************************************
 * Comparison and ostream operators
 ******************************************************************************/

/**
 * KeyValuePair ostream operator
 */
template <typename Key, typename Value>
std::ostream& operator<<(std::ostream& os, const cub::KeyValuePair<Key, Value> &val)
{
    os << '(' << CoutCast(val.key) << ',' << CoutCast(val.value) << ')';
    return os;
}


/******************************************************************************
 * Comparison and ostream operators for CUDA vector types
 ******************************************************************************/

/**
 * Vector1 overloads
 */
#define CUB_VEC_OVERLOAD_1(T, BaseT)                        \
    /* Ostream output */                                    \
    std::ostream& operator<<(                               \
        std::ostream& os,                                   \
        const T& val)                                       \
    {                                                       \
        os << '(' << CoutCast(val.x) << ')';                \
        return os;                                          \
    }                                                       \
    /* Inequality */                                        \
    __host__ __device__ __forceinline__ bool operator!=(    \
        const T &a,                                         \
        const T &b)                                         \
    {                                                       \
        return (a.x != b.x);                                \
    }                                                       \
    /* Equality */                                          \
    __host__ __device__ __forceinline__ bool operator==(    \
        const T &a,                                         \
        const T &b)                                         \
    {                                                       \
        return (a.x == b.x);                                \
    }                                                       \
    /* Test initialization */                               \
    __host__ __device__ __forceinline__ void InitValue(GenMode gen_mode, T &value, int index = 0)   \
    {                                                       \
        InitValue(gen_mode, value.x, index);                \
    }                                                       \
    /* Max */                                               \
    __host__ __device__ __forceinline__ bool operator>(     \
        const T &a,                                         \
        const T &b)                                         \
    {                                                       \
        return (a.x > b.x);                                 \
    }                                                       \
    /* Min */                                               \
    __host__ __device__ __forceinline__ bool operator<(     \
        const T &a,                                         \
        const T &b)                                         \
    {                                                       \
        return (a.x < b.x);                                 \
    }                                                       \
    /* Summation (non-reference addends for VS2003 -O3 warpscan workaround */                       \
    __host__ __device__ __forceinline__ T operator+(        \
        T a,                                                \
        T b)                                                \
    {                                                       \
        T retval = make_##T(a.x + b.x);                     \
        return retval;                                      \
    }                                                       \
    namespace cub {                                         \
    template<>                                              \
    struct NumericTraits<T>                                 \
    {                                                       \
        static const Category CATEGORY = NOT_A_NUMBER;      \
        enum {                                              \
            PRIMITIVE       = false,                        \
            NULL_TYPE       = false,                        \
        };                                                  \
        static T Max()                                      \
        {                                                   \
            T retval = {                                    \
                NumericTraits<BaseT>::Max()};               \
            return retval;                                  \
        }                                                   \
        static T Lowest()                                   \
        {                                                   \
            T retval = {                                    \
                NumericTraits<BaseT>::Lowest()};            \
            return retval;                                  \
        }                                                   \
    };                                                      \
    } /* namespace std */



/**
 * Vector2 overloads
 */
#define CUB_VEC_OVERLOAD_2(T, BaseT)                        \
    /* Ostream output */                                    \
    std::ostream& operator<<(                               \
        std::ostream& os,                                   \
        const T& val)                                       \
    {                                                       \
        os << '('                                           \
            << CoutCast(val.x) << ','                       \
            << CoutCast(val.y) << ')';                      \
        return os;                                          \
    }                                                       \
    /* Inequality */                                        \
    __host__ __device__ __forceinline__ bool operator!=(    \
        const T &a,                                         \
        const T &b)                                         \
    {                                                       \
        return (a.x != b.x) ||                              \
            (a.y != b.y);                                   \
    }                                                       \
    /* Equality */                                          \
    __host__ __device__ __forceinline__ bool operator==(    \
        const T &a,                                         \
        const T &b)                                         \
    {                                                       \
        return (a.x == b.x) &&                              \
            (a.y == b.y);                                   \
    }                                                       \
    /* Test initialization */                               \
    __host__ __device__ __forceinline__ void InitValue(GenMode gen_mode, T &value, int index = 0)   \
    {                                                       \
        InitValue(gen_mode, value.x, index);                \
        InitValue(gen_mode, value.y, index);                \
    }                                                       \
    /* Max */                                               \
    __host__ __device__ __forceinline__ bool operator>(     \
        const T &a,                                         \
        const T &b)                                         \
    {                                                       \
        if (a.x > b.x) return true; else if (b.x > a.x) return false;   \
        return a.y > b.y;                                               \
    }                                                       \
    /* Min */                                               \
    __host__ __device__ __forceinline__ bool operator<(     \
        const T &a,                                         \
        const T &b)                                         \
    {                                                       \
        if (a.x < b.x) return true; else if (b.x < a.x) return false;   \
        return a.y < b.y;                                               \
    }                                                       \
    /* Summation (non-reference addends for VS2003 -O3 warpscan workaround */                                         \
    __host__ __device__ __forceinline__ T operator+(        \
        T a,                                         \
        T b)                                         \
    {                                                       \
        T retval = make_##T(                                        \
            a.x + b.x,                                      \
            a.y + b.y);                                     \
        return retval;                                      \
    }                                                       \
    namespace cub {                                         \
    template<>                                              \
    struct NumericTraits<T>                                 \
    {                                                       \
        static const Category CATEGORY = NOT_A_NUMBER;      \
        enum {                                              \
            PRIMITIVE       = false,                        \
            NULL_TYPE       = false,                        \
        };                                                  \
        static T Max()                                      \
        {                                                   \
            T retval = {                                    \
                NumericTraits<BaseT>::Max(),                \
                NumericTraits<BaseT>::Max()};               \
            return retval;                                  \
        }                                                   \
        static T Lowest()                                   \
        {                                                   \
            T retval = {                                    \
                NumericTraits<BaseT>::Lowest(),             \
                NumericTraits<BaseT>::Lowest()};            \
            return retval;                                  \
        }                                                   \
    };                                                      \
    } /* namespace cub */



/**
 * Vector3 overloads
 */
#define CUB_VEC_OVERLOAD_3(T, BaseT)                        \
    /* Ostream output */                                    \
    std::ostream& operator<<(                               \
        std::ostream& os,                                   \
        const T& val)                                       \
    {                                                       \
        os << '('                                           \
            << CoutCast(val.x) << ','                       \
            << CoutCast(val.y) << ','                       \
            << CoutCast(val.z) << ')';                      \
        return os;                                          \
    }                                                       \
    /* Inequality */                                        \
    __host__ __device__ __forceinline__ bool operator!=(    \
        const T &a,                                         \
        const T &b)                                         \
    {                                                       \
        return (a.x != b.x) ||                              \
            (a.y != b.y) ||                                 \
            (a.z != b.z);                                   \
    }                                                       \
    /* Equality */                                          \
    __host__ __device__ __forceinline__ bool operator==(    \
        const T &a,                                         \
        const T &b)                                         \
    {                                                       \
        return (a.x == b.x) &&                              \
            (a.y == b.y) &&                                 \
            (a.z == b.z);                                   \
    }                                                       \
    /* Test initialization */                               \
    __host__ __device__ __forceinline__ void InitValue(GenMode gen_mode, T &value, int index = 0)   \
    {                                                       \
        InitValue(gen_mode, value.x, index);                \
        InitValue(gen_mode, value.y, index);                \
        InitValue(gen_mode, value.z, index);                \
    }                                                       \
    /* Max */                                               \
    __host__ __device__ __forceinline__ bool operator>(     \
        const T &a,                                         \
        const T &b)                                         \
    {                                                       \
        if (a.x > b.x) return true; else if (b.x > a.x) return false;   \
        if (a.y > b.y) return true; else if (b.y > a.y) return false;   \
        return a.z > b.z;                                               \
    }                                                       \
    /* Min */                                               \
    __host__ __device__ __forceinline__ bool operator<(     \
        const T &a,                                         \
        const T &b)                                         \
    {                                                       \
        if (a.x < b.x) return true; else if (b.x < a.x) return false;   \
        if (a.y < b.y) return true; else if (b.y < a.y) return false;   \
        return a.z < b.z;                                               \
    }                                                       \
    /* Summation (non-reference addends for VS2003 -O3 warpscan workaround */                                         \
    __host__ __device__ __forceinline__ T operator+(        \
        T a,                                                \
        T b)                                                \
    {                                                       \
        T retval = make_##T(                                        \
            a.x + b.x,                                      \
            a.y + b.y,                                      \
            a.z + b.z);                                     \
        return retval;                                      \
    }                                                       \
    namespace cub {                                         \
    template<>                                              \
    struct NumericTraits<T>                                 \
    {                                                       \
        static const Category CATEGORY = NOT_A_NUMBER;      \
        enum {                                              \
            PRIMITIVE       = false,                        \
            NULL_TYPE       = false,                        \
        };                                                  \
        static T Max()                                      \
        {                                                   \
            T retval = {                                    \
                NumericTraits<BaseT>::Max(),                \
                NumericTraits<BaseT>::Max(),                \
                NumericTraits<BaseT>::Max()};               \
            return retval;                                  \
        }                                                   \
        static T Lowest()                                   \
        {                                                   \
            T retval = {                                    \
                NumericTraits<BaseT>::Lowest(),             \
                NumericTraits<BaseT>::Lowest(),             \
                NumericTraits<BaseT>::Lowest()};            \
            return retval;                                  \
        }                                                   \
    };                                                      \
    } /* namespace cub */


/**
 * Vector4 overloads
 */
#define CUB_VEC_OVERLOAD_4(T, BaseT)                        \
    /* Ostream output */                                    \
    std::ostream& operator<<(                               \
        std::ostream& os,                                   \
        const T& val)                                       \
    {                                                       \
        os << '('                                           \
            << CoutCast(val.x) << ','                       \
            << CoutCast(val.y) << ','                       \
            << CoutCast(val.z) << ','                       \
            << CoutCast(val.w) << ')';                      \
        return os;                                          \
    }                                                       \
    /* Inequality */                                        \
    __host__ __device__ __forceinline__ bool operator!=(    \
        const T &a,                                         \
        const T &b)                                         \
    {                                                       \
        return (a.x != b.x) ||                              \
            (a.y != b.y) ||                                 \
            (a.z != b.z) ||                                 \
            (a.w != b.w);                                   \
    }                                                       \
    /* Equality */                                          \
    __host__ __device__ __forceinline__ bool operator==(    \
        const T &a,                                         \
        const T &b)                                         \
    {                                                       \
        return (a.x == b.x) &&                              \
            (a.y == b.y) &&                                 \
            (a.z == b.z) &&                                 \
            (a.w == b.w);                                   \
    }                                                       \
    /* Test initialization */                               \
    __host__ __device__ __forceinline__ void InitValue(GenMode gen_mode, T &value, int index = 0)   \
    {                                                       \
        InitValue(gen_mode, value.x, index);                \
        InitValue(gen_mode, value.y, index);                \
        InitValue(gen_mode, value.z, index);                \
        InitValue(gen_mode, value.w, index);                \
    }                                                       \
    /* Max */                                               \
    __host__ __device__ __forceinline__ bool operator>(     \
        const T &a,                                         \
        const T &b)                                         \
    {                                                       \
        if (a.x > b.x) return true; else if (b.x > a.x) return false;   \
        if (a.y > b.y) return true; else if (b.y > a.y) return false;   \
        if (a.z > b.z) return true; else if (b.z > a.z) return false;   \
        return a.w > b.w;                                               \
    }                                                       \
    /* Min */                                               \
    __host__ __device__ __forceinline__ bool operator<(     \
        const T &a,                                         \
        const T &b)                                         \
    {                                                       \
        if (a.x < b.x) return true; else if (b.x < a.x) return false;   \
        if (a.y < b.y) return true; else if (b.y < a.y) return false;   \
        if (a.z < b.z) return true; else if (b.z < a.z) return false;   \
        return a.w < b.w;                                               \
    }                                                       \
    /* Summation (non-reference addends for VS2003 -O3 warpscan workaround */                                         \
    __host__ __device__ __forceinline__ T operator+(        \
        T a,                                                \
        T b)                                                \
    {                                                       \
        T retval = make_##T(                                        \
            a.x + b.x,                                      \
            a.y + b.y,                                      \
            a.z + b.z,                                      \
            a.w + b.w);                                     \
        return retval;                                      \
    }                                                       \
    namespace cub {                                         \
    template<>                                              \
    struct NumericTraits<T>                                 \
    {                                                       \
        static const Category CATEGORY = NOT_A_NUMBER;      \
        enum {                                              \
            PRIMITIVE       = false,                        \
            NULL_TYPE       = false,                        \
        };                                                  \
        static T Max()                                      \
        {                                                   \
            T retval = {                                    \
                NumericTraits<BaseT>::Max(),                \
                NumericTraits<BaseT>::Max(),                \
                NumericTraits<BaseT>::Max(),                \
                NumericTraits<BaseT>::Max()};               \
            return retval;                                  \
        }                                                   \
        static T Lowest()                                   \
        {                                                   \
            T retval = {                                    \
                NumericTraits<BaseT>::Lowest(),             \
                NumericTraits<BaseT>::Lowest(),             \
                NumericTraits<BaseT>::Lowest(),             \
                NumericTraits<BaseT>::Lowest()};            \
            return retval;                                  \
        }                                                   \
    };                                                      \
    } /* namespace cub */

/**
 * All vector overloads
 */
#define CUB_VEC_OVERLOAD(COMPONENT_T, BaseT)                    \
    CUB_VEC_OVERLOAD_1(COMPONENT_T##1, BaseT)                   \
    CUB_VEC_OVERLOAD_2(COMPONENT_T##2, BaseT)                   \
    CUB_VEC_OVERLOAD_3(COMPONENT_T##3, BaseT)                   \
    CUB_VEC_OVERLOAD_4(COMPONENT_T##4, BaseT)

/**
 * Define for types
 */
CUB_VEC_OVERLOAD(char, char)
CUB_VEC_OVERLOAD(short, short)
CUB_VEC_OVERLOAD(int, int)
CUB_VEC_OVERLOAD(long, long)
CUB_VEC_OVERLOAD(longlong, long long)
CUB_VEC_OVERLOAD(uchar, unsigned char)
CUB_VEC_OVERLOAD(ushort, unsigned short)
CUB_VEC_OVERLOAD(uint, unsigned int)
CUB_VEC_OVERLOAD(ulong, unsigned long)
CUB_VEC_OVERLOAD(ulonglong, unsigned long long)
CUB_VEC_OVERLOAD(float, float)
CUB_VEC_OVERLOAD(double, double)


//---------------------------------------------------------------------
// Complex data type TestFoo
//---------------------------------------------------------------------

/**
 * TestFoo complex data type
 */
struct TestFoo
{
    long long   x;
    int         y;
    short       z;
    char        w;

    // Factory
    static __host__ __device__ __forceinline__ TestFoo MakeTestFoo(long long x, int y, short z, char w)
    {
        TestFoo retval = {x, y, z, w};
        return retval;
    }

    // Assignment from int operator
    __host__ __device__ __forceinline__ TestFoo& operator =(int b)
    {
        x = b;
        y = b;
        z = b;
        w = b;
        return *this;
    }

    // Summation operator
    __host__ __device__ __forceinline__ TestFoo operator+(const TestFoo &b) const
    {
        return MakeTestFoo(x + b.x, y + b.y, z + b.z, w + b.w);
    }

    // Inequality operator
    __host__ __device__ __forceinline__ bool operator !=(const TestFoo &b) const
    {
        return (x != b.x) || (y != b.y) || (z != b.z) || (w != b.w);
    }

    // Equality operator
    __host__ __device__ __forceinline__ bool operator ==(const TestFoo &b) const
    {
        return (x == b.x) && (y == b.y) && (z == b.z) && (w == b.w);
    }

    // Less than operator
    __host__ __device__ __forceinline__ bool operator <(const TestFoo &b) const
    {
        if (x < b.x) return true; else if (b.x < x) return false;
        if (y < b.y) return true; else if (b.y < y) return false;
        if (z < b.z) return true; else if (b.z < z) return false;
        return w < b.w;
    }

    // Greater than operator
    __host__ __device__ __forceinline__ bool operator >(const TestFoo &b) const
    {
        if (x > b.x) return true; else if (b.x > x) return false;
        if (y > b.y) return true; else if (b.y > y) return false;
        if (z > b.z) return true; else if (b.z > z) return false;
        return w > b.w;
    }

};

/**
 * TestFoo ostream operator
 */
std::ostream& operator<<(std::ostream& os, const TestFoo& val)
{
    os << '(' << val.x << ',' << val.y << ',' << val.z << ',' << CoutCast(val.w) << ')';
    return os;
}

/**
 * TestFoo test initialization
 */
__host__ __device__ __forceinline__ void InitValue(GenMode gen_mode, TestFoo &value, int index = 0)
{
    InitValue(gen_mode, value.x, index);
    InitValue(gen_mode, value.y, index);
    InitValue(gen_mode, value.z, index);
    InitValue(gen_mode, value.w, index);
}


/// numeric_limits<TestFoo> specialization
namespace cub {
template<>
struct NumericTraits<TestFoo>
{
    static const Category CATEGORY = NOT_A_NUMBER;
    enum {
        PRIMITIVE       = false,
        NULL_TYPE       = false,
    };
    static TestFoo Max()
    {
        return TestFoo::MakeTestFoo(
            NumericTraits<long long>::Max(),
            NumericTraits<int>::Max(),
            NumericTraits<short>::Max(),
            NumericTraits<char>::Max());
    }

    static TestFoo Lowest()
    {
        return TestFoo::MakeTestFoo(
            NumericTraits<long long>::Lowest(),
            NumericTraits<int>::Lowest(),
            NumericTraits<short>::Lowest(),
            NumericTraits<char>::Lowest());
    }
};
} // namespace cub


//---------------------------------------------------------------------
// Complex data type TestBar (with optimizations for fence-free warp-synchrony)
//---------------------------------------------------------------------

/**
 * TestBar complex data type
 */
struct TestBar
{
    long long       x;
    int             y;

    // Constructor
    __host__ __device__ __forceinline__ TestBar() : x(0), y(0)
    {}

    // Constructor
    __host__ __device__ __forceinline__ TestBar(int b) : x(b), y(b)
    {}

    // Constructor
    __host__ __device__ __forceinline__ TestBar(long long x, int y) : x(x), y(y)
    {}

    // Assignment from int operator
    __host__ __device__ __forceinline__ TestBar& operator =(int b)
    {
        x = b;
        y = b;
        return *this;
    }

    // Summation operator
    __host__ __device__ __forceinline__ TestBar operator+(const TestBar &b) const
    {
        return TestBar(x + b.x, y + b.y);
    }

    // Inequality operator
    __host__ __device__ __forceinline__ bool operator !=(const TestBar &b) const
    {
        return (x != b.x) || (y != b.y);
    }

    // Equality operator
    __host__ __device__ __forceinline__ bool operator ==(const TestBar &b) const
    {
        return (x == b.x) && (y == b.y);
    }

    // Less than operator
    __host__ __device__ __forceinline__ bool operator <(const TestBar &b) const
    {
        if (x < b.x) return true; else if (b.x < x) return false;
        return y < b.y;
    }

    // Greater than operator
    __host__ __device__ __forceinline__ bool operator >(const TestBar &b) const
    {
        if (x > b.x) return true; else if (b.x > x) return false;
        return y > b.y;
    }

};


/**
 * TestBar ostream operator
 */
std::ostream& operator<<(std::ostream& os, const TestBar& val)
{
    os << '(' << val.x << ',' << val.y << ')';
    return os;
}

/**
 * TestBar test initialization
 */
__host__ __device__ __forceinline__ void InitValue(GenMode gen_mode, TestBar &value, int index = 0)
{
    InitValue(gen_mode, value.x, index);
    InitValue(gen_mode, value.y, index);
}

/// numeric_limits<TestBar> specialization
namespace cub {
template<>
struct NumericTraits<TestBar>
{
    static const Category CATEGORY = NOT_A_NUMBER;
    enum {
        PRIMITIVE       = false,
        NULL_TYPE       = false,
    };
    static TestBar Max()
    {
        return TestBar(
            NumericTraits<long long>::Max(),
            NumericTraits<int>::Max());
    }

    static TestBar Lowest()
    {
        return TestBar(
            NumericTraits<long long>::Lowest(),
            NumericTraits<int>::Lowest());
    }
};
} // namespace cub


/******************************************************************************
 * Helper routines for list comparison and display
 ******************************************************************************/


/**
 * Compares the equivalence of two arrays
 */
template <typename S, typename T, typename OffsetT>
int CompareResults(T* computed, S* reference, OffsetT len, bool verbose = true)
{
    for (OffsetT i = 0; i < len; i++)
    {
        if (computed[i] != reference[i])
        {
            if (verbose) std::cout << "INCORRECT: [" << i << "]: "
                << CoutCast(computed[i]) << " != "
                << CoutCast(reference[i]);
            return 1;
        }
    }
    return 0;
}


/**
 * Compares the equivalence of two arrays
 */
template <typename OffsetT>
int CompareResults(float* computed, float* reference, OffsetT len, bool verbose = true)
{
    for (OffsetT i = 0; i < len; i++)
    {
        if (computed[i] != reference[i])
        {
            float difference = std::abs(computed[i]-reference[i]);
            float fraction = difference / std::abs(reference[i]);

            if (fraction > 0.0001)
            {
                if (verbose) std::cout << "INCORRECT: [" << i << "]: "
                    << "(computed) " << CoutCast(computed[i]) << " != "
                    << CoutCast(reference[i]) << " (difference:" << difference << ", fraction: " << fraction << ")";
                return 1;
            }
        }
    }
    return 0;
}


/**
 * Compares the equivalence of two arrays
 */
template <typename OffsetT>
int CompareResults(cub::NullType* computed, cub::NullType* reference, OffsetT len, bool verbose = true)
{
    return 0;
}

/**
 * Compares the equivalence of two arrays
 */
template <typename OffsetT>
int CompareResults(double* computed, double* reference, OffsetT len, bool verbose = true)
{
    for (OffsetT i = 0; i < len; i++)
    {
        if (computed[i] != reference[i])
        {
            double difference = std::abs(computed[i]-reference[i]);
            double fraction = difference / std::abs(reference[i]);

            if (fraction > 0.0001)
            {
                if (verbose) std::cout << "INCORRECT: [" << i << "]: "
                    << CoutCast(computed[i]) << " != "
                    << CoutCast(reference[i]) << " (difference:" << difference << ", fraction: " << fraction << ")";
                return 1;
            }
        }
    }
    return 0;
}


/**
 * Verify the contents of a device array match those
 * of a host array
 */
int CompareDeviceResults(
    cub::NullType *h_reference,
    cub::NullType *d_data,
    size_t num_items,
    bool verbose = true,
    bool display_data = false)
{
    return 0;
}

/**
 * Verify the contents of a device array match those
 * of a host array
 */
template <typename S, typename OffsetT>
int CompareDeviceResults(
    S *h_reference,
    cub::DiscardOutputIterator<OffsetT> d_data,
    size_t num_items,
    bool verbose = true,
    bool display_data = false)
{
    return 0;
}

/**
 * Verify the contents of a device array match those
 * of a host array
 */
template <typename S, typename T>
int CompareDeviceResults(
    S *h_reference,
    T *d_data,
    size_t num_items,
    bool verbose = true,
    bool display_data = false)
{
    // Allocate array on host
    T *h_data = (T*) malloc(num_items * sizeof(T));

    // Copy data back
    cudaMemcpy(h_data, d_data, sizeof(T) * num_items, cudaMemcpyDeviceToHost);

    // Display data
    if (display_data)
    {
        printf("Reference:\n");
        for (int i = 0; i < int(num_items); i++)
        {
            std::cout << CoutCast(h_reference[i]) << ", ";
        }
        printf("\n\nComputed:\n");
        for (int i = 0; i < int(num_items); i++)
        {
            std::cout << CoutCast(h_data[i]) << ", ";
        }
        printf("\n\n");
    }

    // Check
    int retval = CompareResults(h_data, h_reference, num_items, verbose);

    // Cleanup
    if (h_data) free(h_data);

    return retval;
}


/**
 * Verify the contents of a device array match those
 * of a device array
 */
template <typename T>
int CompareDeviceDeviceResults(
    T *d_reference,
    T *d_data,
    size_t num_items,
    bool verbose = true,
    bool display_data = false)
{
    // Allocate array on host
    T *h_reference = (T*) malloc(num_items * sizeof(T));
    T *h_data = (T*) malloc(num_items * sizeof(T));

    // Copy data back
    cudaMemcpy(h_reference, d_reference, sizeof(T) * num_items, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_data, d_data, sizeof(T) * num_items, cudaMemcpyDeviceToHost);

    // Display data
    if (display_data) {
        printf("Reference:\n");
        for (int i = 0; i < num_items; i++)
        {
            std::cout << CoutCast(h_reference[i]) << ", ";
        }
        printf("\n\nComputed:\n");
        for (int i = 0; i < num_items; i++)
        {
            std::cout << CoutCast(h_data[i]) << ", ";
        }
        printf("\n\n");
    }

    // Check
    int retval = CompareResults(h_data, h_reference, num_items, verbose);

    // Cleanup
    if (h_reference) free(h_reference);
    if (h_data) free(h_data);

    return retval;
}


/**
 * Print the contents of a host array
 */
void DisplayResults(
    cub::NullType   *h_data,
    size_t          num_items)
{}


/**
 * Print the contents of a host array
 */
template <typename InputIteratorT>
void DisplayResults(
    InputIteratorT h_data,
    size_t num_items)
{
    // Display data
    for (int i = 0; i < int(num_items); i++)
    {
        std::cout << CoutCast(h_data[i]) << ", ";
    }
    printf("\n");
}


/**
 * Print the contents of a device array
 */
template <typename T>
void DisplayDeviceResults(
    T *d_data,
    size_t num_items)
{
    // Allocate array on host
    T *h_data = (T*) malloc(num_items * sizeof(T));

    // Copy data back
    cudaMemcpy(h_data, d_data, sizeof(T) * num_items, cudaMemcpyDeviceToHost);

    DisplayResults(h_data, num_items);

    // Cleanup
    if (h_data) free(h_data);
}


/******************************************************************************
 * Segment descriptor generation
 ******************************************************************************/

/**
 * Initialize segments
 */
void InitializeSegments(
    int     num_items,
    int     num_segments,
    int     *h_segment_offsets,
    bool    verbose = false)
{
    if (num_segments <= 0)
        return;

    unsigned int expected_segment_length = (num_items + num_segments - 1) / num_segments;
    int offset = 0;
    for (int i = 0; i < num_segments; ++i)
    {
        h_segment_offsets[i] = offset;

        unsigned int segment_length = RandomValue((expected_segment_length * 2) + 1);
        offset += segment_length;
        offset = CUB_MIN(offset, num_items);
    }
    h_segment_offsets[num_segments] = num_items;

    if (verbose)
    {
        printf("Segment offsets: ");
        DisplayResults(h_segment_offsets, num_segments + 1);
    }
}


/******************************************************************************
 * Timing
 ******************************************************************************/


struct CpuTimer
{
#if defined(_WIN32) || defined(_WIN64)

    LARGE_INTEGER ll_freq;
    LARGE_INTEGER ll_start;
    LARGE_INTEGER ll_stop;

    CpuTimer()
    {
        QueryPerformanceFrequency(&ll_freq);
    }

    void Start()
    {
        QueryPerformanceCounter(&ll_start);
    }

    void Stop()
    {
        QueryPerformanceCounter(&ll_stop);
    }

    float ElapsedMillis()
    {
        double start = double(ll_start.QuadPart) / double(ll_freq.QuadPart);
        double stop  = double(ll_stop.QuadPart) / double(ll_freq.QuadPart);

        return float((stop - start) * 1000);
    }

#else

    rusage start;
    rusage stop;

    void Start()
    {
        getrusage(RUSAGE_SELF, &start);
    }

    void Stop()
    {
        getrusage(RUSAGE_SELF, &stop);
    }

    float ElapsedMillis()
    {
        float sec = stop.ru_utime.tv_sec - start.ru_utime.tv_sec;
        float usec = stop.ru_utime.tv_usec - start.ru_utime.tv_usec;

        return (sec * 1000) + (usec / 1000);
    }

#endif
};

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float ElapsedMillis()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};
