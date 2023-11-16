#include "gpu_match.cuh"
#include <cuda.h>

#define UNROLL_SIZE(l) (l > 0 ? UNROLL : 1)

#define LANEID (threadIdx.x % WARP_SIZE)
#define PEAK_CLK (float)1410000 // A100
#define ELAPSED_TIME(start) (clock() - start)/PEAK_CLK // in ms
#define TIMEOUT 10 // timeout

namespace STMatch
{
	struct StealingArgs
	{
		int *idle_warps;
		int *idle_warps_count;
		int *global_mutex;
		int *local_mutex;
		CallStack *global_callstack;
		Queue *queue;
	};

	__forceinline__ __device__ void lock(int *mutex)
	{
		while (atomicCAS((int *)mutex, 0, 1) != 0)
		{
		}
	}
	__forceinline__ __device__ void unlock(int *mutex)
	{
		atomicExch((int *)mutex, 0);
	}
	
	// target_stk is the warp being stolen
	__device__ bool trans_layer(CallStack &_target_stk, CallStack &_cur_stk, Pattern *_pat, int _k, int ratio = 2)
	{
		if (_target_stk.level <= _k)
			return false;

		int num_left_task = _target_stk.slot_size[_k] - (_target_stk.iter[_k] + 1);
		if (num_left_task <= 0)
			return false;

		int stealed_start_idx_in_target = _target_stk.iter[_k] + 1 + num_left_task / ratio;


		// lvl 0 to _k - 1 (inclusive)
		_cur_stk.slot_storage[0][_target_stk.iter[0]] = _target_stk.slot_storage[0][_target_stk.iter[0]];
		_cur_stk.slot_storage[0][_target_stk.iter[0] + JOB_CHUNK_SIZE] = _target_stk.slot_storage[0][_target_stk.iter[0] + JOB_CHUNK_SIZE];
		for (int i = 1; i < _k; i++)
		{
			_cur_stk.slot_storage[i][_target_stk.iter[i]] = _target_stk.slot_storage[i][_target_stk.iter[i]];
		}

		// lvl _k (inclusive)
		int loop_end = _k == 0 ? JOB_CHUNK_SIZE * 2 : _target_stk.slot_size[_k];
		for (int t = 0; t < loop_end; t++)
		{
			_cur_stk.slot_storage[_k][t] = _target_stk.slot_storage[_k][t];
		}

		for (int l = 0; l < _k; l++)
		{
			_cur_stk.iter[l] = _target_stk.iter[l];
			_cur_stk.slot_size[l] = _target_stk.iter[l] + 1;
		}

		_cur_stk.slot_size[_k] = _target_stk.slot_size[_k];
		_cur_stk.iter[_k] = stealed_start_idx_in_target;
		_target_stk.slot_size[_k] = stealed_start_idx_in_target;

		// copy
		for (int l = _k + 1; l < _pat->nnodes - 1; l++)
		{
			_cur_stk.iter[l] = 0;
			_cur_stk.slot_size[l] = 0;
		}
		_cur_stk.iter[_pat->nnodes - 1] = 0;
		_cur_stk.slot_size[_pat->nnodes - 1] = 0;
	
		_cur_stk.level = _k + 1;
		return true;
	}
	
	__device__ bool trans_skt(CallStack *_all_stk, CallStack *_cur_stk, Pattern *pat, StealingArgs *_stealing_args)
	{

		int max_left_task = 0;
		int stk_idx = -1;
		int at_level = -1;

		for (int level = 0; level < STOP_LEVEL; level++)
		{
			for (int i = 0; i < NWARPS_PER_BLOCK; i++)
			{
				if (i == threadIdx.x / WARP_SIZE)
					continue;
				lock(&(_stealing_args->local_mutex[i]));

				int left_task = _all_stk[i].slot_size[level] - (_all_stk[i].iter[level] + 1);
				if (left_task > max_left_task)
				{
					max_left_task = left_task;
					stk_idx = i;
					at_level = level;
				}
				unlock(&(_stealing_args->local_mutex[i]));
			}
			if (stk_idx != -1)
				break;
		}

		if (stk_idx != -1)
		{
			bool res;
			lock(&(_stealing_args->local_mutex[threadIdx.x / WARP_SIZE]));
			lock(&(_stealing_args->local_mutex[stk_idx]));
			res = trans_layer(_all_stk[stk_idx], *_cur_stk, pat, at_level);

			unlock(&(_stealing_args->local_mutex[threadIdx.x / WARP_SIZE]));
			unlock(&(_stealing_args->local_mutex[stk_idx]));
			return res;
		}
		return false;
	}

	__forceinline__ __device__ graph_node_t path(CallStack *stk, Pattern *pat, int level)
	{
		if (level > 0)
			return stk->slot_storage[level][stk->iter[level]];
		else
		{
			return stk->slot_storage[0][stk->iter[0] + (level + 1) * JOB_CHUNK_SIZE]; // level=-1 or 0
		}
	}

	__forceinline__ __device__ graph_node_t *path_address(CallStack *stk, Pattern *pat, int level)
	{
		if (level > 0)
			return &(stk->slot_storage[level][stk->iter[level]]);
		else
		{
			return &(stk->slot_storage[0][stk->iter[0] + (level + 1) * JOB_CHUNK_SIZE]); // -1 or 0
		}
	}

	typedef struct
	{
		graph_node_t *set1, *set2, *res;
		graph_node_t set1_size, set2_size, *res_size;
		graph_node_t ub;
		bitarray32 label;
		Graph *g;
		int num_sets;
		bool cached;
		int level;
		Pattern *pat;
	} Arg_t;

	template <typename DATA_T, typename SIZE_T>
	__forceinline__ __device__ bool bsearch_exist(DATA_T *set2, SIZE_T set2_size, DATA_T target)
	{
		if (set2_size <= 0)
			return false;
		int mid;
		int low = 0;
		int high = set2_size - 1;
		while (low <= high)
		{
			mid = (low + high) / 2;
			if (target == set2[mid])
			{
				return true;
			}
			else if (target > set2[mid])
			{
				low = mid + 1;
			}
			else
			{
				high = mid - 1;
			}
		}
		return false;
	}

	template <typename DATA_T, typename SIZE_T>
	__forceinline__ __device__
		SIZE_T
		upper_bound(DATA_T *set2, SIZE_T set2_size, DATA_T target)
	{
		int i, step;
		int low = 0;
		while (set2_size > 0)
		{
			i = low;
			step = set2_size / 2;
			i += step;
			if (target > set2[i])
			{
				low = ++i;
				set2_size -= step + 1;
			}
			else
			{
				set2_size = step;
			}
		}
		return low;
	}

	__forceinline__ __device__ void prefix_sum(int *_input, int input_size)
	{

		int thid = threadIdx.x % WARP_SIZE;
		int offset = 1;
		int last_element = _input[input_size - 1];
		// build sum in place up the tree
		for (int d = (WARP_SIZE >> 1); d > 0; d >>= 1)
		{
			if (thid < d)
			{
				int ai = offset * (2 * thid + 1) - 1;
				int bi = offset * (2 * thid + 2) - 1;
				_input[bi] += _input[ai];
			}
			offset <<= 1;
		}
		if (thid == 0)
		{
			_input[WARP_SIZE - 1] = 0;
		} // clear the last element
		  // traverse down tree & build scan
		for (int d = 1; d < WARP_SIZE; d <<= 1)
		{
			offset >>= 1;
			if (thid < d)
			{
				int ai = offset * (2 * thid + 1) - 1;
				int bi = offset * (2 * thid + 2) - 1;
				int t = _input[ai];
				_input[ai] = _input[bi];
				_input[bi] += t;
			}
		}
		__syncwarp();

		if (thid >= input_size - 1)
			_input[thid + 1] = _input[input_size - 1] + last_element;
	}

	__forceinline__ __device__ int scanIndex(bool pred)
	{
		unsigned int bits = __ballot_sync(0xFFFFFFFF, pred);
		unsigned int mask = 0xFFFFFFFF >> (31 - threadIdx.x % WARP_SIZE);
		int index = __popc(mask & bits) - pred; // to get exclusive sum subtract pred
		return index;
	}

	__forceinline__ __device__ void compute_intersection(Arg_t *arg, CallStack *stk, bool last_round, bool check_validity)
	{
		int res_length = 0;
		int actual_lvl = arg->level + 1;
		bool pred;
		int target;
		int cur_label = arg->pat->vertex_labels[actual_lvl];

		for (int i = 0; i < arg->set2_size; i += WARP_SIZE)
		{
			pred = false;
			int il = i + LANEID;
			if (il < arg->set2_size)
			{
				pred = true;
				target = arg->set2[il];

				if (check_validity)
				{
					if (!LABELED)
					{
						// if unlabeled, check automorphism
						for (int k = 0; k < arg->pat->condition_cnt[actual_lvl]; ++k)
						{
							int cond = arg->pat->condition_order[actual_lvl * PAT_SIZE * 2 + 2 * k];
							int cond_lvl = arg->pat->condition_order[actual_lvl * PAT_SIZE * 2 + 2 * k + 1];
							int cond_vertex_M = path(stk, arg->pat, cond_lvl - 1);
							if (cond == CondOperator::LESS_THAN) {
								if (cond_vertex_M <= target) {
									pred = false;
									break;
								}
							}
							else if (cond == CondOperator::LARGER_THAN) {
								if (cond_vertex_M >= target) {
									pred = false;
									break;
								}
							}
							else if (cond == CondOperator::NON_EQUAL) {
								if (cond_vertex_M == target) {
									pred = false;
									break;
								}
							}
						}
					}
					else
					{
						// if LABELED, check label
						if (arg->g->vertex_label[target] != cur_label)
						{
							pred = false;
						}
						// STMatch does no check 
						if (pred)
						{
							for (int k = -1; k < arg->level; ++k)
							{
								int cond_vertex_M = path(stk, arg->pat, k);
								if (cond_vertex_M == target) {
									pred = false;
									break;
								}
							}
						}
					}
				}
				if (pred) pred = bsearch_exist(arg->set1, arg->set1_size, target);
			}
			int loc = scanIndex(pred) + res_length;
			if ((arg->level < arg->pat->nnodes - 2 && pred) || ((arg->level == arg->pat->nnodes - 2) && !last_round && pred))
				arg->res[loc] = target;
			if (threadIdx.x % WARP_SIZE == 31) // last lane's loc+pred is number of items found in this scan
				res_length = loc + pred;
			res_length = __shfl_sync(0xFFFFFFFF, res_length, 31);
		}
		*arg->res_size = res_length;
	}

	__forceinline__ __device__ void arr_copy(Arg_t *arg, CallStack *stk)
	{
		int res_length = 0;
		int actual_lvl = arg->level + 1;
		bool pred;
		int target;
		int cur_label = arg->pat->vertex_labels[actual_lvl];
		int BN = arg->pat->backward_neighbors[actual_lvl][0];

		graph_node_t t = path(stk, arg->pat, BN - 1);
		int num_neighbor = (graph_node_t)(arg->g->rowptr[t + 1] - arg->g->rowptr[t]);
		int *neighbors = &(arg->g->colidx[arg->g->rowptr[t]]);

		for (int i = 0; i < num_neighbor; i += WARP_SIZE)
		{
			// if unlabeled, check automorphism
			pred = false;
			int il = i + LANEID;
			if (il < num_neighbor)
			{
				pred = true;
				target = neighbors[il];
				if (!LABELED) {
					for (int k = 0; k < arg->pat->condition_cnt[actual_lvl]; ++k)
					{
						int cond = arg->pat->condition_order[actual_lvl * PAT_SIZE * 2 + 2 * k];
						int cond_lvl = arg->pat->condition_order[actual_lvl * PAT_SIZE * 2 + 2 * k + 1];
						int cond_vertex_M = path(stk, arg->pat, cond_lvl - 1);
						if (cond == CondOperator::LESS_THAN) {
							if (cond_vertex_M <= target) {
								pred = false;
								break;
							}
						}
						else if (cond == CondOperator::LARGER_THAN) {
							if (cond_vertex_M >= target) {
								pred = false;
								break;
							}
						}
						else if (cond == CondOperator::NON_EQUAL) {
							if (cond_vertex_M == target) {
								pred = false;
								break;
							}
						}
					}
				} 
				else
				{
					if (arg->g->vertex_label[target] != cur_label)
					{
						pred = false;
					}
					// STMatch does no check 
					if (pred)
					{
						for (int k = -1; k < arg->level; ++k)
						{
							int cond_vertex_M = path(stk, arg->pat, k);
							if (cond_vertex_M == target) {
								pred = false;
								break;
							}
						}
					}
				}
			}
			int loc = scanIndex(pred) + res_length;
			if (arg->level < arg->pat->nnodes - 2 && pred)
				stk->slot_storage[arg->level][loc] = target;
			if (threadIdx.x % WARP_SIZE == 31) // last lane's loc+pred is number of items found in this scan
				res_length = loc + pred;
			res_length = __shfl_sync(0xFFFFFFFF, res_length, 31);
		}
		stk->slot_size[arg->level] = res_length;
	}

	__forceinline__ __device__ void arr_copy_shared(Arg_t *arg, CallStack *stk)
	{
		int res_length = 0;
		int actual_lvl = arg->level + 1;
		bool pred;
		int target;
		int cur_label = arg->pat->vertex_labels[actual_lvl];
		
		graph_node_t dep = arg->pat->shared_lvl[actual_lvl];
		assert(dep != -1);
		int num_neighbor = stk->slot_size[dep -1];
		int *neighbors = stk->slot_storage[dep - 1];

		for (int i = 0; i < num_neighbor; i += WARP_SIZE)
		{
			// if unlabeled, check automorphism
			pred = false;
			int il = i + LANEID;
			if (il < num_neighbor)
			{
				pred = true;
				target = neighbors[il];
				if (!LABELED) {
					for (int k = 0; k < arg->pat->condition_cnt[actual_lvl]; ++k)
					{
						int cond = arg->pat->condition_order[actual_lvl * PAT_SIZE * 2 + 2 * k];
						int cond_lvl = arg->pat->condition_order[actual_lvl * PAT_SIZE * 2 + 2 * k + 1];
						int cond_vertex_M = path(stk, arg->pat, cond_lvl - 1);
						if (cond == CondOperator::LESS_THAN) {
							if (cond_vertex_M <= target) {
								pred = false;
								break;
							}
						}
						else if (cond == CondOperator::LARGER_THAN) {
							if (cond_vertex_M >= target) {
								pred = false;
								break;
							}
						}
						else if (cond == CondOperator::NON_EQUAL) {
							if (cond_vertex_M == target) {
								pred = false;
								break;
							}
						}
					}
				} 
				else
				{
					if (arg->g->vertex_label[target] != cur_label)
					{
						pred = false;
					}
					// STMatch does no check 
					if (pred)
					{
						for (int k = -1; k < arg->level; ++k)
						{
							int cond_vertex_M = path(stk, arg->pat, k);
							if (cond_vertex_M == target) {
								pred = false;
								break;
							}
						}
					}
				}
			}
			int loc = scanIndex(pred) + res_length;
			if (arg->level < arg->pat->nnodes - 2 && pred)
				stk->slot_storage[arg->level][loc] = target;
			if (threadIdx.x % WARP_SIZE == 31) // last lane's loc+pred is number of items found in this scan
				res_length = loc + pred;
			res_length = __shfl_sync(0xFFFFFFFF, res_length, 31);
		}
		stk->slot_size[arg->level] = res_length;
	}

	__forceinline__ __device__ void get_job(Graph *g, Pattern *pat, CallStack *stk, JobQueue *q)
	// __forceinline__ __device__ void get_job(JobQueue *q, graph_node_t &cur_pos, graph_node_t &njobs)
	{
		// lock(&(q->mutex));
		// cur_pos = q->cur;
		// q->cur += JOB_CHUNK_SIZE;
		// if (q->cur > q->length)
		// 	q->cur = q->length;
		// njobs = q->cur - cur_pos;
		// unlock(&(q->mutex));

		// cur_pos = atomicAdd(&q->cur, JOB_CHUNK_SIZE);
		// if (cur_pos < q->length) {
		// 	int tmp = cur_pos + JOB_CHUNK_SIZE;
		// 	if (tmp > q->length) 
		// 		tmp = q->length;
		// 	njobs = tmp - cur_pos;
		// }
        // else
        // {
	    //     atomicAdd(&q->cur, -JOB_CHUNK_SIZE);
        //     njobs = 0;
        // }

		
		int cnt;
		while (true)
		{
			unsigned long long cur_pos = atomicAdd(&q->cur, JOB_CHUNK_SIZE);
			cnt = 0;
			if (cur_pos < g->nedges)
			{
				long end = cur_pos + JOB_CHUNK_SIZE;
				if (end > g->nedges)
					end = g->nedges;
				for (long i = cur_pos; i < end; ++i)
				{
					graph_node_t c = g->colidx[i];
					graph_node_t r = g->src_vtx[i];
					
					if ((!LABELED && pat->partial_ori[0][0] == 1 && r < c) || LABELED || pat->partial_ori[0][0] != 1)
          			{
            			if (!LABELED || (g->vertex_label[r] == pat->vertex_labels[0] && g->vertex_label[c] == pat->vertex_labels[1]) )
						{
							if (g->rowptr[r + 1] - g->rowptr[r] >= pat->degree[0] && g->rowptr[c + 1] - g->rowptr[c] >= pat->degree[1]) {
								bool valid = false;
								for (graph_edge_t d = g->rowptr[c]; d < g->rowptr[c + 1]; d++) {
									graph_node_t v = g->colidx[d];
									if (g->rowptr[v + 1] - g->rowptr[v] >= pat->degree[2]) {
										valid = true;
										break;
									}
								}
								if (valid)
								{
									stk->slot_storage[0][cnt] = r;
									stk->slot_storage[0][JOB_CHUNK_SIZE + cnt] = c;
									cnt++;
								}
							}
						}
					}
				}
				stk->slot_size[0] = cnt;

				if (cnt > 0)
					break;
			} else {
				atomicAdd(&q->cur, -JOB_CHUNK_SIZE);
				stk->slot_size[0] = 0;
				break;
			}
		}
	}

	__device__ void extend(Graph *g, Pattern *pat, CallStack *stk, JobQueue *q, pattern_node_t level, long &start_clk, 
							StealingArgs *_stealing_args)
	{

		__shared__ Arg_t arg[NWARPS_PER_BLOCK];

		int wid = threadIdx.x / WARP_SIZE;

		if (level == 0)
		{
			// TODO: change to warp

			graph_node_t cur_job, njobs;
			stk->stealed_task = false;
			
			if (threadIdx.x % WARP_SIZE == 0)
			{
				int x, y, z;
				bool ret = _stealing_args->queue->dequeue(x, y, z);
				if (ret) {
					stk->slot_storage[0][0] = x;
					stk->slot_storage[0][JOB_CHUNK_SIZE] = y;
					stk->slot_size[0] = 1;

					if (z != DeletionMarker<int>::val - 1)
					{
						level = 1;
						stk->slot_storage[1][0] = z;
						stk->slot_size[1] = 1;

						stk->stealed_task = true;
					}
				}
				else
				{
					get_job(g, pat, stk, q);

					// get_job(q, cur_job, njobs);

					// for (size_t i = 0; i < njobs; i++)
                    // {
                    //     for (int j = 0; j < 2; j++)
                    //     {
                    //         stk->slot_storage[0][i + JOB_CHUNK_SIZE * j] = (q->q[cur_job + i].nodes)[j]; // matches of 2 nodes are saved at level 0
                    //     }
                    // }
                    // stk->slot_size[0] = njobs;
				}
			}
			__syncwarp();
			start_clk = clock64();
		}
		else
		{
			arg[wid].g = g;
			arg[wid].level = level;
			arg[wid].pat = pat;

			int actual_lvl = level + 1;

			bool last_round;
			
			int dep = pat->shared_lvl[actual_lvl];

			if (dep == -1 ||
				(dep == 2 && stk->stealed_task))
			{
				if (pat->num_BN[actual_lvl] == 0)
					assert(false);
				else if (pat->num_BN[actual_lvl] == 1)
				{
					arr_copy(&arg[wid], stk);
				}
				else
				{
					int BN = pat->backward_neighbors[actual_lvl][0];
					graph_node_t t = path(stk, pat, BN - 1);
					int i_min = 0;
					int t_min = t;
					int min_neighbor = (graph_node_t)(g->rowptr[t + 1] - g->rowptr[t]);

					for (int i = 1; i < pat->num_BN[actual_lvl]; ++i)
					{
						BN = pat->backward_neighbors[actual_lvl][i];
						t = path(stk, pat, BN - 1);
						int neighbor_cnt = (graph_node_t)(g->rowptr[t + 1] - g->rowptr[t]);
						if (neighbor_cnt < min_neighbor)
						{
							i_min = i;
							t_min = t;
							min_neighbor = neighbor_cnt;
						}
					}
					// arr_copy(stk->slot_storage[level], &g->colidx[g->rowptr[t_min]], min_neighbor);
					// stk->slot_size[level] = min_neighbor;

					if (i_min != 0)
					{
						BN = pat->backward_neighbors[actual_lvl][0];
						t = path(stk, pat, BN - 1);
						int* neighbor = &g->colidx[g->rowptr[t]];
						int neighbor_cnt = (graph_node_t)(g->rowptr[t + 1] - g->rowptr[t]);
						arg[wid].set1 = neighbor;
						arg[wid].set1_size = neighbor_cnt;
						arg[wid].set2 = &g->colidx[g->rowptr[t_min]];
						arg[wid].set2_size = min_neighbor;
						arg[wid].res = stk->slot_storage[level];
						arg[wid].res_size = &(stk->slot_size[level]);
						last_round = (pat->num_BN[actual_lvl] == 2) ? true : false;
						compute_intersection(&arg[wid], stk, last_round, true);

						for (int i = 1; i < pat->num_BN[actual_lvl]; ++i)
						{
							if (i == i_min) continue;
							last_round = (i == pat->num_BN[actual_lvl] - 1) || (i == pat->num_BN[actual_lvl] - 2 && i_min == pat->num_BN[actual_lvl] - 1);
							BN = pat->backward_neighbors[actual_lvl][i];
							t = path(stk, pat, BN - 1);
							int* neighbor = &g->colidx[g->rowptr[t]];
							int neighbor_cnt = (graph_node_t)(g->rowptr[t + 1] - g->rowptr[t]);
							arg[wid].set1 = neighbor;
							arg[wid].set1_size = neighbor_cnt;
							arg[wid].set2 = stk->slot_storage[level];
							arg[wid].set2_size = stk->slot_size[level];
							arg[wid].res = stk->slot_storage[level];
							arg[wid].res_size = &(stk->slot_size[level]);
							compute_intersection(&arg[wid], stk, last_round, false);
						}
					}
					else // i_min = 0 
					{
						BN = pat->backward_neighbors[actual_lvl][1];
						t = path(stk, pat, BN - 1);
						int* neighbor = &g->colidx[g->rowptr[t]];
						int neighbor_cnt = (graph_node_t)(g->rowptr[t + 1] - g->rowptr[t]);
						arg[wid].set1 = neighbor;
						arg[wid].set1_size = neighbor_cnt;
						arg[wid].set2 = &g->colidx[g->rowptr[t_min]];
						arg[wid].set2_size = min_neighbor;
						arg[wid].res = stk->slot_storage[level];
						arg[wid].res_size = &(stk->slot_size[level]);
						last_round = (pat->num_BN[actual_lvl] == 2) ? true : false;
						compute_intersection(&arg[wid], stk, last_round, true);

						for (int i = 2; i < pat->num_BN[actual_lvl]; ++i)
						{
							if (i == i_min) continue;
							last_round = (i == pat->num_BN[actual_lvl] - 1) || (i == pat->num_BN[actual_lvl] - 2 && i_min == pat->num_BN[actual_lvl] - 1);
							BN = pat->backward_neighbors[actual_lvl][i];
							t = path(stk, pat, BN - 1);
							int* neighbor = &g->colidx[g->rowptr[t]];
							int neighbor_cnt = (graph_node_t)(g->rowptr[t + 1] - g->rowptr[t]);
							arg[wid].set1 = neighbor;
							arg[wid].set1_size = neighbor_cnt;
							arg[wid].set2 = stk->slot_storage[level];
							arg[wid].set2_size = stk->slot_size[level];
							arg[wid].res = stk->slot_storage[level];
							arg[wid].res_size = &(stk->slot_size[level]);
							compute_intersection(&arg[wid], stk, last_round, false);
						}
					}
				}
			} 
			else 
			{
				if (pat->num_BN_sh[actual_lvl] == 0)
				{
					arr_copy_shared(&arg[wid], stk);
				}
				else 
				{
					int BN = pat->backward_neighbors_sh[actual_lvl][0];
					graph_node_t t = path(stk, pat, BN - 1);
					int i_min = 0;
					int t_min = t;
					int min_neighbor = (graph_node_t)(g->rowptr[t + 1] - g->rowptr[t]);

					int dep = pat->shared_lvl[actual_lvl];
					int* neighbor = stk->slot_storage[dep - 1];
					int neighbor_cnt = stk->slot_size[dep - 1];
					arg[wid].set1 = &g->colidx[g->rowptr[t_min]];
					arg[wid].set1_size = min_neighbor;
					arg[wid].set2 = neighbor;
					arg[wid].set2_size = neighbor_cnt;
					arg[wid].res = stk->slot_storage[level];
					arg[wid].res_size = &(stk->slot_size[level]);
					last_round = (pat->num_BN_sh[actual_lvl] == 1) ? true : false;
					compute_intersection(&arg[wid], stk, last_round, true);

					for (int i = 1; i < pat->num_BN_sh[actual_lvl]; ++i)
					{
						if (i == i_min) continue;
						last_round = (i == pat->num_BN_sh[actual_lvl] - 1) || (i == pat->num_BN_sh[actual_lvl] - 2 && i_min == pat->num_BN_sh[actual_lvl] - 1);
						BN = pat->backward_neighbors_sh[actual_lvl][i];
						t = path(stk, pat, BN - 1);
						int* neighbor = &g->colidx[g->rowptr[t]];
						int neighbor_cnt = (graph_node_t)(g->rowptr[t + 1] - g->rowptr[t]);
						arg[wid].set1 = neighbor;
						arg[wid].set1_size = neighbor_cnt;
						arg[wid].set2 = stk->slot_storage[level];
						arg[wid].set2_size = stk->slot_size[level];
						arg[wid].res = stk->slot_storage[level];
						arg[wid].res_size = &(stk->slot_size[level]);
						compute_intersection(&arg[wid], stk, last_round, false);
					}
				}
			}
		}
		stk->iter[level] = 0;
	}

	
	__forceinline__ __device__ void respond_across_block(int level, CallStack *stk, Pattern *pat, StealingArgs *_stealing_args)
	{
		if (level > 0 && level <= DETECT_LEVEL)
		{
			if (threadIdx.x % WARP_SIZE == 0)
			{
				int at_level = -1;
				int left_task = 0;
				for (int l = 0; l < level; l++)
				{
					left_task = stk->slot_size[l] - stk->iter[l] - 1;
					if (left_task > 0)
					{
						at_level = l;
						break;
					}
				}
				if (at_level != -1)
				{
					for (int b = 0; b < GRID_DIM; b++)
					{
						if (b == blockIdx.x)
							continue;
						if (atomicCAS(&(_stealing_args->global_mutex[b]), 0, 1) == 0)
						{
							if (atomicAdd(&_stealing_args->idle_warps[b], 0) == 0xFFFFFFFF)
							{
								__threadfence();

								trans_layer(*stk, _stealing_args->global_callstack[b * NWARPS_PER_BLOCK], pat, at_level, INT_MAX);
								__threadfence();

								atomicSub(_stealing_args->idle_warps_count, NWARPS_PER_BLOCK);
								atomicExch(&_stealing_args->idle_warps[b], 0);

								atomicExch(&(_stealing_args->global_mutex[b]), 0);
								break;
							}
							atomicExch(&(_stealing_args->global_mutex[b]), 0);
						}
					}
				}
			}
			__syncwarp();
		}
	}
	

	__device__ void match(Graph *g, Pattern *pat,
						  CallStack *stk, JobQueue *q, size_t *count, StealingArgs *_stealing_args, long &start_clk)
	{

		pattern_node_t &level = stk->level;

		while (true)
		{
			// if (threadIdx.x % WARP_SIZE == 0)
			// {
			// 	lock(&(_stealing_args->local_mutex[threadIdx.x / WARP_SIZE]));
			// }
			// __syncwarp();

			if (level < pat->nnodes - 2)
			{
				// if (true)
				// {
				// 	respond_across_block(level, stk, pat, _stealing_args);
				// }

				if (stk->slot_size[level] == 0)
				{
					extend(g, pat, stk, q, level, start_clk, _stealing_args);
					if (level == 0 && stk->slot_size[0] == 0)
					{
						// if (threadIdx.x % WARP_SIZE == 0)
						// 	unlock(&(_stealing_args->local_mutex[threadIdx.x / WARP_SIZE]));
						// __syncwarp();
						break;
					}
				}

				int is_timeout;
				if (LANEID == 0)
					is_timeout = level < STOP_LEVEL && ELAPSED_TIME(start_clk) > TIMEOUT;
				is_timeout = __shfl_sync(0xFFFFFFFF, is_timeout, 0);

				if (stk->iter[level] < stk->slot_size[level] && !is_timeout)
				// if (stk->iter[level] < stk->slot_size[level])
				{
					if (threadIdx.x % WARP_SIZE == 0)
						level++;
					__syncwarp();
				}
				else if (stk->iter[level] < stk->slot_size[level] && is_timeout)
				{
					int enqueue_succ;
					if (LANEID == 0)
					{
						for(; stk->iter[level] < stk->slot_size[level]; stk->iter[level]++)
						{
							int x, y, z;
							x = path(stk, pat, -1);
							y = path(stk, pat, 0);
							if (level == 1)
								z = path(stk, pat, 1);
							else
								z = DeletionMarker<int>::val - 1;
							enqueue_succ = _stealing_args->queue->enqueue(x, y, z);
							if (!enqueue_succ) break;
						}
					}
					enqueue_succ = __shfl_sync(0xFFFFFFFF, enqueue_succ, 0);
					if (enqueue_succ)
					{
						stk->slot_size[level] = 0;
						stk->iter[level] = 0;
						if (level > 0)
						{
							if (threadIdx.x % WARP_SIZE == 0)
								level--;
							if (threadIdx.x % WARP_SIZE == 0)
								stk->iter[level]++;
							__syncwarp();
						}
					} else { // means queue is full, reset timer, do some work
						start_clk = clock64();
					}
				}
				else
				{
					stk->slot_size[level] = 0;
					stk->iter[level] = 0;
					if (level > 0)
					{
						if (threadIdx.x % WARP_SIZE == 0)
							level--;
						if (threadIdx.x % WARP_SIZE == 0)
							stk->iter[level]++;
						__syncwarp();

						if (level == 0)
							start_clk = clock64();
					}
				}
			}
			else if (level == pat->nnodes - 2)
			{
				extend(g, pat, stk, q, level, start_clk, _stealing_args);

				if (LANEID == 0)
				{
					*count += stk->slot_size[level];
				}
				__syncwarp();
				stk->slot_size[level] = 0;

				if (threadIdx.x % WARP_SIZE == 0)
					level--;
				if (threadIdx.x % WARP_SIZE == 0)
					stk->iter[level]++;
				__syncwarp();
			}
			__syncwarp();
			// if (threadIdx.x % WARP_SIZE == 0)
			// 	unlock(&(_stealing_args->local_mutex[threadIdx.x / WARP_SIZE]));
			// __syncwarp();
		}
	}

	__global__ void _parallel_match(Graph *dev_graph, Pattern *dev_pattern,
									CallStack *dev_callstack, JobQueue *job_queue, size_t *res,
									int *idle_warps, int *idle_warps_count, int *global_mutex,
									Queue *queue)
	{
		queue->init();

		__shared__ Graph graph;
		__shared__ Pattern pat;
		__shared__ CallStack stk[NWARPS_PER_BLOCK];
		__shared__ size_t count[NWARPS_PER_BLOCK];
		__shared__ bool stealed[NWARPS_PER_BLOCK];
		__shared__ int mutex_this_block[NWARPS_PER_BLOCK];

		__shared__ StealingArgs stealing_args;
		stealing_args.idle_warps = idle_warps;
		stealing_args.idle_warps_count = idle_warps_count;
		stealing_args.global_mutex = global_mutex;
		stealing_args.local_mutex = mutex_this_block;
		stealing_args.global_callstack = dev_callstack;

		stealing_args.queue = queue;

		int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
		int global_wid = global_tid / WARP_SIZE;
		int local_wid = threadIdx.x / WARP_SIZE;

		if (threadIdx.x == 0)
		{
			graph = *dev_graph;
			pat = *dev_pattern;
		}
		__syncthreads();

		if (threadIdx.x % WARP_SIZE == 0)
		{
			stk[local_wid] = dev_callstack[global_wid];
		}
		__syncwarp();

		if (threadIdx.x == 0)
		{
			for(int i=0;i<NWARPS_PER_BLOCK;++i)
			{
				mutex_this_block[i] = 0;
				count[i] = 0;
			}
    	}
     	__syncthreads();

		auto start = clock64();

		while (true)
		{
			long start_clk = clock64();
			match(&graph, &pat, &stk[local_wid], job_queue, &count[local_wid], &stealing_args, start_clk);
			__syncwarp();

			stealed[local_wid] = false;

			
			// if (true)
			// {
			// 	if (threadIdx.x % WARP_SIZE == 0)
			// 	{
			// 		stealed[local_wid] = trans_skt(stk, &stk[local_wid], &pat, &stealing_args);
			// 	}
			// 	__syncwarp();
			// }

			
			// if (true)
			// {
			// 	if (!stealed[local_wid])
			// 	{

			// 		__syncthreads();

			// 		if (threadIdx.x % WARP_SIZE == 0)
			// 		{

			// 			atomicAdd(stealing_args.idle_warps_count, 1);

			// 			lock(&(stealing_args.global_mutex[blockIdx.x]));

			// 			atomicOr(&stealing_args.idle_warps[blockIdx.x], (1 << local_wid));

			// 			unlock(&(stealing_args.global_mutex[blockIdx.x]));

			// 			while ((atomicAdd(stealing_args.idle_warps_count, 0) < NWARPS_TOTAL) && (atomicAdd(&stealing_args.idle_warps[blockIdx.x], 0) & (1 << local_wid)))
			// 				;

			// 			if (atomicAdd(stealing_args.idle_warps_count, 0) < NWARPS_TOTAL)
			// 			{

			// 				__threadfence();
			// 				if (local_wid == 0)
			// 				{
			// 					stk[local_wid] = (stealing_args.global_callstack[blockIdx.x * NWARPS_PER_BLOCK]);
			// 				}
			// 				stealed[local_wid] = true;
			// 			}
			// 			else
			// 			{
			// 				stealed[local_wid] = false;
			// 			}
			// 		}
			// 		__syncthreads();
			// 	}
			// }
			

			if (!stealed[local_wid])
			{
				break;
			}
		}

		auto stop = clock64();

		if (threadIdx.x % WARP_SIZE == 0)
		{
			res[global_wid] = count[local_wid];
			// printf("%d\t%ld\t%d\t%d\n", blockIdx.x, stop - start, stealed[local_wid], local_wid);
			// printf("%ld\n", stop - start);
		}

		// if(threadIdx.x % WARP_SIZE == 0)
		//   printf("%d\t%d\t%d\n", blockIdx.x, local_wid, mutex_this_block[local_wid]);
	}
}
