#pragma once

#include "graph.h"
#include "pattern.h"
#include "callstack.h"
#include "job_queue.h"
#include "queue.h"

namespace STMatch {

template <typename MemoryManagerType>
__global__ void _parallel_match(MemoryManagerType *mm, Graph* dev_graph, Pattern* dev_pattern, 
                            CallStack* dev_callstack, JobQueue* job_queue,  size_t* res,
                            int* idle_warps, int* idle_warps_count, int* global_lock,
                            Queue *queue);

template <typename MemoryManagerType>
__global__ void allocate_memory(MemoryManagerType *mm, CallStack *dev_callstack);
}