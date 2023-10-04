#pragma once

#include "config.h"

namespace STMatch
{

// /**
//  * 3 prefix composites into a 64 bit long integer
//  * 1bit  21bits     21bits     21bits  = 64 bit long integer
//  *  |-|----------|----------|----------|
//  *   0     x          y           z
//  */
// __forceinline__ __device__ unsigned long long set(int x, int y, int z)
// {
//     unsigned long long composite_prefix = ((unsigned long long)x << 42) | ((unsigned long long)y << 21) | (unsigned long long)z;
//     return composite_prefix;
// }
// __forceinline__ __device__ void get(unsigned long long prefix, int &x, int &y, int &z)
// {
//     x = prefix >> 42;
//     y = (prefix >> 21) & 0x1FFFFF;
//     z = prefix & 0x1FFFFF;  // 0b 0001 1111 1111 1111 1111 1111
// }

template <typename DataType>
struct DeletionMarker
{
	static constexpr void* val{ nullptr };
};

template <>
struct DeletionMarker<int>
{
	static constexpr int val {(int)0xFFFFFFFF};
};

template<>
struct DeletionMarker<unsigned long long>
{
    static constexpr unsigned long long val {0xFFFFFFFFFFFFFFFF};
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
            queue_[i] = DeletionMarker<int>::val;
        }
    }
    // x - y - z - ....
	__forceinline__ __device__ bool enqueue(int x, int y, int z)
    {
        int fill = atomicAdd(&count_, 3);
        if (fill < size_)
        {
            unsigned int pos = atomicAdd(&back_, 3) % size_;
            while (atomicCAS(queue_ + pos, DeletionMarker<int>::val, x) != DeletionMarker<int>::val)
                __nanosleep(10);
            while (atomicCAS(queue_ + pos + 1, DeletionMarker<int>::val, y) != DeletionMarker<int>::val)
                __nanosleep(10);
            while (atomicCAS(queue_ + pos + 2, DeletionMarker<int>::val, z) != DeletionMarker<int>::val)
                __nanosleep(10);
            return true;
        }
        else
        {
            // assert(false);
            // __trap(); // no space to enqueue
            atomicSub(&count_, 3);
            return false;
        }
    }
    // (x1, y1, z1) (x1, y1, z2), (x1, y2, z1) ...
	__forceinline__ __device__ bool dequeue(int &x, int &y, int &z)
    {
        int readable = atomicSub(&count_, 3);
        if (readable <= 0)
        {
            atomicAdd(&count_, 3);
            return false;
        }
        unsigned int pos = atomicAdd(&front_, 3) % size_;
        while ((x = atomicExch(queue_ + pos, DeletionMarker<int>::val)) == DeletionMarker<int>::val)
            __nanosleep(10);
        while ((y = atomicExch(queue_ + pos + 1, DeletionMarker<int>::val)) == DeletionMarker<int>::val)
            __nanosleep(10); 
        while ((z = atomicExch(queue_ + pos + 2, DeletionMarker<int>::val)) == DeletionMarker<int>::val)
            __nanosleep(10);
        return true;
    }

	int* queue_;
	int count_{ 0 };
	unsigned int front_{ 0 };
	unsigned int back_{ 0 };
	int size_{ 0 };
};
    
} // namespace STMatch