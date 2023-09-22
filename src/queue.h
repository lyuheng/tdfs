#pragma once

#include "config.h"

namespace STMatch
{

/**
 * 3 prefix composites into a 64 bit long integer
 * 1bit  21bits     21bits     21bits  = 64 bit long integer
 *  |-|----------|----------|----------|
 *   0     x          y           z
 */
__forceinline__ __device__ long set(int x, int y, int z)
{
    long composite_prefix = (x << 42) | (y << 21) | z;
    return composite_prefix;
}
__forceinline__ __device__ void get(long prefix, int &x, int &y, int &z)
{
    x = prefix >> 42;
    y = (prefix >> 21) | & 0x1FFFFF;
    z = prefix & 0x1FFFFF;  // 0b 0001 1111 1111 1111 1111 1111
}

template <typename DataType>
struct DeletionMarker
{
	static constexpr void* val{ nullptr };
};

template <>
struct DeletionMarker<int>
{
	static constexpr int val { 0xFFFFFFFF };
};

template<>
struct DeletionMarker<long>
{
    static constexpr long val {0xFFFFFFFFFFFFFFFF};
};

class Queue
{
public:
    __forceinline__ __host__ __device__ void resetQueue()
    {
        count_ = 0;
        front_ = 0;
        back_ = 0;
    }

	__forceinline__ __device__ void init()
    {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size_; i += blockDim.x * gridDim.x)
        {
            queue_[i] = DeletionMarker<long>::val;
        }
    }

	__forceinline__ __device__ bool enqueue(long &p)
    {
        int fill = atomicAdd(&count_, 1);
        if (fill < size_)
        {
            unsigned int pos = atomicAdd(&back_, 1) % size_;
            while (atomicCAS(queue_ + pos, DeletionMarker<long>::val, p) != DeletionMarker<Prefix>::val)
                __nanosleep(10);
            return true;
        }
        else
        {
            __trap(); // no space to enqueue
            return false;
        }
    }

	__forceinline__ __device__ bool dequeue(long& element)
    {
        int readable = atomicSub(&count_, 1);
        if (readable <= 0)
        {
            atomicAdd(&count_, 1);
            return false;
        }
        unsigned int pos = atomicAdd(&front_, 1) % size_;
        while ((element = atomicExch(queue_ + pos, DeletionMarker<long>::val)) == DeletionMarker<long>::val)
            __nanosleep(10);
        return true;
    }

	long* queue_;
	int count_{ 0 };
	unsigned int front_{ 0 };
	unsigned int back_{ 0 };
	int size_{ 0 };
};
    
} // namespace STMatch