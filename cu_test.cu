#include <string>
#include <iostream>
#include "src/gpu_match.cuh"

using namespace std;
using namespace STMatch;

#define TIMEOUT_QUEUE_CAP 1'000'000

int main(int argc, char* argv[]) {

  cudaSetDevice(0);


  STMatch::GraphPreprocessor g(argv[1]);

  std::cout << "Before PatternPreprocessor" << std::endl;
  
  STMatch::PatternPreprocessor p(argv[2]);
  
  std::cout << "Before build_src_vtx" << std::endl;

  g.build_src_vtx(p);

  std::cout << "after build_src_vtx" << std::endl;


  std::cout << "conditions: " << std::endl;
  for (int i = 0; i < p.order_.size(); i++) 
  {
      std::cout << i << ": ";
      for (int j = 0; j < p.order_[i].size(); j++)
          std::cout << GetCondOperatorString(p.order_[i][j].first) << "(" << p.order_[i][j].second << "), ";
      std::cout << std::endl;
  }

  // copy graph and pattern to GPU global memory
  Graph* gpu_graph = g.to_gpu();
  Pattern* gpu_pattern = p.to_gpu();
  JobQueue* gpu_queue = JobQueuePreprocessor(g.g, p).to_gpu();
  // JobQueue* gpu_queue = nullptr;
  CallStack* gpu_callstack;

  // allocate the callstack for all warps in global memory
  graph_node_t* slot_storage;
  cudaMalloc(&slot_storage, sizeof(graph_node_t) * NWARPS_TOTAL * MAX_SLOT_NUM * GRAPH_DEGREE);
  //cout << "global memory usage: " << sizeof(graph_node_t) * NWARPS_TOTAL * MAX_SLOT_NUM * UNROLL * GRAPH_DEGREE / 1024.0 / 1024 / 1024 << " GB" << endl;

  std::vector<CallStack> stk(NWARPS_TOTAL);

  for (int i = 0; i < NWARPS_TOTAL; i++) {
    auto& s = stk[i];
    memset(s.iter, 0, sizeof(s.iter));
    memset(s.slot_size, 0, sizeof(s.slot_size));
    s.slot_storage = (graph_node_t(*)[GRAPH_DEGREE])((char*)slot_storage + i * sizeof(graph_node_t) * MAX_SLOT_NUM * GRAPH_DEGREE);
  }
  cudaMalloc(&gpu_callstack, NWARPS_TOTAL * sizeof(CallStack));
  cudaMemcpy(gpu_callstack, stk.data(), sizeof(CallStack) * NWARPS_TOTAL, cudaMemcpyHostToDevice);

  size_t* gpu_res;
  cudaMalloc(&gpu_res, sizeof(size_t) * NWARPS_TOTAL);
  cudaMemset(gpu_res, 0, sizeof(size_t) * NWARPS_TOTAL);
  size_t* res = new size_t[NWARPS_TOTAL];

  int* idle_warps;
  cudaMalloc(&idle_warps, sizeof(int) * GRID_DIM);
  cudaMemset(idle_warps, 0, sizeof(int) * GRID_DIM);

  int* idle_warps_count;
  cudaMalloc(&idle_warps_count, sizeof(int));
  cudaMemset(idle_warps_count, 0, sizeof(int));

  int* global_mutex;
  cudaMalloc(&global_mutex, sizeof(int) * GRID_DIM);
  cudaMemset(global_mutex, 0, sizeof(int) * GRID_DIM);

  bool* stk_valid;
  cudaMalloc(&stk_valid, sizeof(bool) * GRID_DIM);
  cudaMemset(stk_valid, 0, sizeof(bool) * GRID_DIM);

  int* gpu_timeout_queue_space;
  cudaMalloc(&gpu_timeout_queue_space, sizeof(int) * TIMEOUT_QUEUE_CAP * (STOP_LEVEL + 1));
  Queue* gpu_timeout_queue;
  cudaMallocManaged(&gpu_timeout_queue, sizeof(Queue));
  gpu_timeout_queue->queue_ = gpu_timeout_queue_space;
  gpu_timeout_queue->size_ = TIMEOUT_QUEUE_CAP * (STOP_LEVEL + 1);
  gpu_timeout_queue->resetQueue();
  
  // timer starts here
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  //cout << "shared memory usage: " << sizeof(Graph) << " " << sizeof(Pattern) << " " << sizeof(JobQueue) << " " << sizeof(CallStack) * NWARPS_PER_BLOCK << " " << NWARPS_PER_BLOCK * 33 * sizeof(int) << " Bytes" << endl;

  _parallel_match << <GRID_DIM, BLOCK_DIM >> > (gpu_graph, gpu_pattern, gpu_callstack, gpu_queue, gpu_res, idle_warps, 
                                              idle_warps_count, global_mutex, gpu_timeout_queue);


  cudaEventRecord(stop);
  
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  //printf("matching time: %f ms\n", milliseconds);

  cudaMemcpy(res, gpu_res, sizeof(size_t) * NWARPS_TOTAL, cudaMemcpyDeviceToHost);

  unsigned long long tot_count = 0;
  for (int i=0; i<NWARPS_TOTAL; i++) tot_count += res[i];

  if(!LABELED) tot_count = tot_count * p.PatternMultiplicity;
  
  printf("%s\t%f\t%llu\n", argv[2], milliseconds, tot_count);
  //cout << "count: " << tot_count << endl;
  return 0;
}
