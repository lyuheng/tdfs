#pragma once

#include "config.h"

namespace STMatch
{

struct Prefix 
{
    int prefix[3];

    bool operator==(const Prefix& p)
    {
        return  prefix[0] == p.prefix[0] && 
                prefix[1] == p.prefix[1] && 
                prefix[2] == p.prefix[2];
    }
};

template <typename DataType>
struct DeletionMarker
{
	static constexpr void* val{ nullptr };
};

template <>
struct DeletionMarker<int>
{
	static constexpr int val{ 0xFFFFFFFF };
};

template<>
struct DeletionMarker<Prefix>
{
    static constexpr Prefix val {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF};
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
            queue_[i] = DeletionMarker<Prefix>::val;
        }
    }

	__forceinline__ __device__ bool enqueue(Prefix &p)
    {
        int fill = atomicAdd(&count_, 1);
        if (fill < size_)
        {
            unsigned int pos = atomicAdd(&back_, 1) % size_;
            while (atomicCAS(queue_ + pos, DeletionMarker<Prefix>::val, p) != DeletionMarker<Prefix>::val)
                __nanosleep(10);
            return true;
        }
        else
        {
            __trap(); // no space to enqueue
            return false;
        }
    }

	__forceinline__ __device__ bool dequeue(Prefix& element)
    {
        int readable = atomicSub(&count_, 1);
        if (readable <= 0)
        {
            atomicAdd(&count_, 1);
            return false;
        }
        unsigned int pos = atomicAdd(&front_, 1) % size_;
        while ((element = atomicExch(queue_ + pos, DeletionMarker<Prefix>::val)) == DeletionMarker<Prefix>::val)
            __nanosleep(10);
        return true;
    }

	Prefix* queue_;
	int count_{ 0 };
	unsigned int front_{ 0 };
	unsigned int back_{ 0 };
	int size_{ 0 };
};
    
} // namespace STMatch