#include <string>
#include <iostream>
#include "src/gpu_match.cuh"

using namespace std;
using namespace STMatch;

#define TIMEOUT_QUEUE_CAP 1'000'000

int main(int argc, char* argv[]) {

  STMatch::GraphPreprocessor g(argv[1]);
  STMatch::PatternPreprocessor p(argv[2]);
  g.build_src_vtx(p);

  
  Graph* gpu_graph[NUM_GPU];
  Pattern* gpu_pattern[NUM_GPU];
  JobQueue* gpu_queue[NUM_GPU];
  CallStack* gpu_callstack[NUM_GPU];
  graph_node_t* slot_storage[NUM_GPU];
  size_t* gpu_res[NUM_GPU];

  int* idle_warps[NUM_GPU];
  int* idle_warps_count[NUM_GPU];
  int* global_mutex[NUM_GPU];
  bool* stk_valid[NUM_GPU];

  int *gpu_timeout_queue_space[NUM_GPU];
  Queue* gpu_timeout_queue[NUM_GPU];

  for(int gpuIdx=0; gpuIdx<NUM_GPU; gpuIdx++){
    cudaSetDevice(gpuIdx);
    
    gpu_graph[gpuIdx] = g.to_gpu();
    gpu_pattern[gpuIdx] = p.to_gpu();
    gpu_queue[gpuIdx] = JobQueuePreprocessor(g.g, p, gpuIdx).to_gpu();
    cudaMalloc(&slot_storage[gpuIdx], sizeof(graph_node_t) *  NWARPS_TOTAL * MAX_SLOT_NUM * GRAPH_DEGREE);
    std::vector<CallStack> stk(NWARPS_TOTAL);
    for (int i = 0; i < NWARPS_TOTAL; i++) {
      auto& s = stk[i];
      memset(s.iter, 0, sizeof(s.iter));
      memset(s.slot_size, 0, sizeof(s.slot_size));
      s.slot_storage = (graph_node_t(*)[GRAPH_DEGREE])((char*)slot_storage[gpuIdx] + i * sizeof(graph_node_t) * MAX_SLOT_NUM * GRAPH_DEGREE);
    }
    cudaMalloc(&gpu_callstack[gpuIdx], NWARPS_TOTAL * sizeof(CallStack));
    cudaMemcpy(gpu_callstack[gpuIdx], stk.data(), sizeof(CallStack) * NWARPS_TOTAL, cudaMemcpyHostToDevice);

    cudaMalloc(&gpu_res[gpuIdx], sizeof(size_t) * NWARPS_TOTAL);
    cudaMemset(gpu_res[gpuIdx], 0, sizeof(size_t) * NWARPS_TOTAL);

    cudaMalloc(&idle_warps[gpuIdx], sizeof(int) * GRID_DIM);
    cudaMemset(idle_warps[gpuIdx], 0, sizeof(int) * GRID_DIM);

    cudaMalloc(&idle_warps_count[gpuIdx], sizeof(int));
    cudaMemset(idle_warps_count[gpuIdx], 0, sizeof(int));

    cudaMalloc(&global_mutex[gpuIdx], sizeof(int) * GRID_DIM);
    cudaMemset(global_mutex[gpuIdx], 0, sizeof(int) * GRID_DIM);

    cudaMalloc(&stk_valid[gpuIdx], sizeof(bool) * GRID_DIM);
    cudaMemset(stk_valid[gpuIdx], 0, sizeof(bool) * GRID_DIM);

    cudaMalloc(&gpu_timeout_queue_space[gpuIdx], sizeof(int) * TIMEOUT_QUEUE_CAP * (STOP_LEVEL + 1));
    cudaMallocManaged(&gpu_timeout_queue[gpuIdx], sizeof(Queue));
    gpu_timeout_queue[gpuIdx]->queue_ = gpu_timeout_queue_space[gpuIdx];
    gpu_timeout_queue[gpuIdx]->size_ = TIMEOUT_QUEUE_CAP * (STOP_LEVEL + 1);
    gpu_timeout_queue[gpuIdx]->resetQueue();

  }
  
  size_t* res = new size_t[NWARPS_TOTAL];
  cudaEvent_t start[NUM_GPU], stop[NUM_GPU];
  float milliseconds[NUM_GPU];
  
//--------------------  
  #pragma omp parallel for num_threads(NUM_GPU)
  for(int i=0; i<NUM_GPU; i++) {
      cudaSetDevice(i);
      cudaEventCreate(&start[i]);
      cudaEventCreate(&stop[i]);
      cudaEventRecord(start[i]);
      _parallel_match << <GRID_DIM, BLOCK_DIM>> > (gpu_graph[i], gpu_pattern[i], gpu_callstack[i], gpu_queue[i], gpu_res[i], 
                                                  idle_warps[i], idle_warps_count[i], global_mutex[i], gpu_timeout_queue[i]);
      cudaEventRecord(stop[i]);
      cudaEventSynchronize(stop[i]);
      cudaEventElapsedTime(&milliseconds[i], start[i], stop[i]);
   }
   //printf("All Finished\n");



  float maxGPU = 0;
  unsigned long long finalCount =0;
  for(int i=0; i<NUM_GPU; i++) {
    cudaMemcpy(res, gpu_res[i], sizeof(size_t) * NWARPS_TOTAL, cudaMemcpyDeviceToHost);
    size_t tot_count = 0;
    for (int j=0; j<NWARPS_TOTAL; j++) {
      tot_count += res[j];
    }
    if(milliseconds[i]>maxGPU) maxGPU = milliseconds[i];
    finalCount+=tot_count;

    //printf("%f\t", milliseconds[i]);
  }

  if(!LABELED) finalCount = finalCount * p.PatternMultiplicity;
  printf("%s\t%f\t%llu\n", argv[2], maxGPU, finalCount);

  return 0;
}
