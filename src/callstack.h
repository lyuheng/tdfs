#pragma once

#include <stdlib.h> 
#include <string.h>
#include <stdio.h>
#include <cuda.h>
#include "config.h"
#include "Ouroboros/include/device/Ouroboros_impl.cuh"
#include "Ouroboros/include/device/MemoryInitialization.cuh"
#include "Ouroboros/include/InstanceDefinitions.cuh"
#include "Ouroboros/include/Utility.h"

#define LARGEST_PAGE_SIZE (1<<13)  //2^13 B = 8KB

namespace STMatch {

static constexpr int countBitShift(unsigned int x)
{
	if (x == 0) return 0;
	int n = 0;
	if (x <= 0x0000FFFF) { n = n + 16; x = x << 16; }
	if (x <= 0x00FFFFFF) { n = n + 8; x = x << 8; }
	if (x <= 0x0FFFFFFF) { n = n + 4; x = x << 4; }
	if (x <= 0x3FFFFFFF) { n = n + 2; x = x << 2; }
	if (x <= 0x7FFFFFFF) { n = n + 1; x = x << 1; }
	return 31 - n;
}

template <typename DataT, typename MemoryManagerType>
struct PageBuffer
{
    static constexpr int ELEMENTS_PER_PAGE = LARGEST_PAGE_SIZE/sizeof(DataT);
    static constexpr int ELEMENTS_BITS = countBitShift(ELEMENTS_PER_PAGE);
    int *page_num; // # page number kept in index_map
    DataT **index_map;

    __forceinline__ __device__ DataT* allocate_new_page(MemoryManagerType *mm) {
        int allocation_size_byte = LARGEST_PAGE_SIZE;
        allocation_size_byte = Ouro::alignment(allocation_size_byte, sizeof(int));
        return reinterpret_cast<DataT *>(mm->malloc(allocation_size_byte)); // cast to DataT*
    }
    __forceinline__ __device__ DataT& operator()(int idx, MemoryManagerType *mm) { 
        // thread-safe
        if (page_num != NULL) {
            // int cur_page_num = atomicAdd(page_num, 0);// *page_num
            int cur_page_num = *page_num;
            int page_id = idx >> ELEMENTS_BITS; // div
            int offset = idx & (ELEMENTS_PER_PAGE - 1); // mod

            // if (page_id > cur_page_num)
            // {
                // printf("idx=%d, cur_page_num=%d\n", idx, cur_page_num);
                // assert(page_id <= cur_page_num);
            // }
            if (page_id == cur_page_num)
            {
                int mask = __activemask();
                int leader = __ffs(mask) - 1;
                if (LANEID == leader)
                {
                    DataT *new_page_addr = allocate_new_page(mm);
                    index_map[cur_page_num] = new_page_addr;
                    // atomicAdd(page_num, 1);
                    *page_num += 1;
                    __threadfence(); // TODO: removal is okay?
                }
                __syncwarp(mask);
            }
            return index_map[page_id][offset];
        }
        else { // page_num == NULL
            return ((DataT *)index_map)[idx];
        }
    }
};

template <typename DataT, typename MemoryManagerType>
struct StackBuffers
{
    using PageBufferT = PageBuffer<DataT, MemoryManagerType>;
    PageBufferT buffers[PAT_SIZE];
    MemoryManagerType *mm;
    __forceinline__ __device__ __host__ DataT& operator()(int slot_id, int offset_id) { 
      return buffers[slot_id](offset_id, mm);
    }
    __forceinline__ __device__ __host__ PageBufferT& operator()(int slot_id) { 
      return buffers[slot_id];
    }
    __forceinline__ __device__ void allocate() {
        int allocation_size_byte = LARGEST_PAGE_SIZE;
        allocation_size_byte = Ouro::alignment(allocation_size_byte, sizeof(int));
        for (int i = 0; i < PAT_SIZE; ++i) {
          DataT *page_addr = reinterpret_cast<DataT *>(mm->malloc(allocation_size_byte));
          buffers[i].index_map[0] = page_addr;
        }
        // if(GTHID == 0)
        //     printf("PageBufferT::ELEMENTS_PER_PAGE = %d\n", PageBufferT::ELEMENTS_PER_PAGE);
    }
    __forceinline__ __device__ void free() {
      for (int i = 0; i < PAT_SIZE; ++i) {
        mm->free(buffers[i].index_map[0]);
        buffers[i].page_num = 0;
      }
    }
};


typedef struct {
  graph_node_t iter[PAT_SIZE];
  graph_node_t uiter[PAT_SIZE];
  graph_node_t slot_size[MAX_SLOT_NUM];
  // graph_node_t (*slot_storage)[GRAPH_DEGREE];
  StackBuffers<graph_node_t, MemoryManagerType> slot_storage;
  pattern_node_t level;
} CallStack;

/*
  void init() {
    memset(path, 0, sizeof(path));
    memset(iter, 0, sizeof(iter));
    memset(slot_size, 0, sizeof(slot_size));
  }
  */
}