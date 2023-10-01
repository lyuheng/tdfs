
#pragma once

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <vector>
#include <map>
#include <string>
#include <set>
#include <unordered_set>
#include <cassert>
#include <utility>
#include "config.h"
#include "../bliss/graph.hh"

namespace STMatch {

enum CondOperator { LESS_THAN = 0, LARGER_THAN, NON_EQUAL, OPERATOR_NONE };


inline std::string GetCondOperatorString(const CondOperator& op) {
  std::string ret = "";
  switch (op) {
    case LESS_THAN:
        ret = "LESS_THAN";
        break;
    case LARGER_THAN:
        ret = "LAGER_THAN";
        break;
    case NON_EQUAL:
        ret = "NON_EQUAL";
        break;
    default:
        break;
  }
  return ret;
}
   

  typedef struct {

    pattern_node_t nnodes = 0;
    int rowptr[PAT_SIZE];
    int degree[PAT_SIZE];
    bitarray32 slot_labels[MAX_SLOT_NUM];
    bitarray32 partial[MAX_SLOT_NUM];
    set_op_t set_ops[MAX_SLOT_NUM];

    int backward_neighbors[PAT_SIZE][PAT_SIZE];
    int num_BN[PAT_SIZE];
    int condition_order[PAT_SIZE * PAT_SIZE * 2];
    int condition_cnt[PAT_SIZE];
    int vertex_labels[PAT_SIZE];
    bitarray32 partial[PAT_SIZE][PAT_SIZE];
  } Pattern;


  struct PatternPreprocessor {

    Pattern pat;

    int PatternMultiplicity;
    int adj_matrix_[PAT_SIZE][PAT_SIZE];
    int vertex_order_[PAT_SIZE];
    int order_map_[PAT_SIZE];
    std::vector<std::vector<int>> L_adj_matrix_;
    std::vector<std::vector<int>> board;

    bitarray32 slot_labels[PAT_SIZE][PAT_SIZE];
    bitarray32 partial[PAT_SIZE][PAT_SIZE];
    set_op_t set_ops[PAT_SIZE][PAT_SIZE];

    std::vector<int> vertex_labels;

    int length[PAT_SIZE];
    int edge[PAT_SIZE][PAT_SIZE];

    typedef std::pair<CondOperator, int> CondType;
    typedef std::vector<std::vector<CondType>> AllCondType;

    AllCondType order_;

    PatternPreprocessor(std::string filename) {
      readfile(filename);
      SetConditions(GetConditions(GetBlissGraph())); // set order_
      get_matching_order();
      get_partial_order();
      get_set_ops();
      propagate_partial_order();
      get_labels();
      convert2oned();

      //std::cout << "Pattern read complete. Pattern size: " << (int)pat.nnodes << std::endl;
    }

    Pattern* to_gpu() {
      Pattern* patcopy;
      cudaMalloc(&patcopy, sizeof(Pattern));
      cudaMemcpy(patcopy, &pat, sizeof(Pattern), cudaMemcpyHostToDevice);
      return patcopy;
    }

    void readfile(std::string& filename) {
      //std::cout << filename << std::endl;

      std::ifstream fin(filename);
      std::string line;
      while (std::getline(fin, line) && (line[0] == '#'));
      pat.nnodes = 0;

      do {
        std::istringstream sin(line);
        char tmp;
        int v;
        int label;
        sin >> tmp >> v >> label;
        if(LABELED){
          vertex_labels.push_back(label);
        }
        else {
          vertex_labels.push_back(1);
        }
        pat.nnodes++;
      } while (std::getline(fin, line) && (line[0] == 'v'));

      memset(adj_matrix_, 0, sizeof(adj_matrix_));
      do {
        std::istringstream sin(line);
        char tmp;
        int v1, v2;
        int label;
        sin >> tmp >> v1 >> v2 >> label;
        adj_matrix_[v1][v2] = label;
        adj_matrix_[v2][v1] = label;
      } while (getline(fin, line));
    }


    // input from dryadic is alreay reordered 
    void get_matching_order() {
      /* int root = 0;
      int max_degree = 0;
      for (int i = 0; i < pat.nnodes; i++) {
        int d = 0;
        for (int j = 0; j < pat.nnodes; j++) {
          if (adj_matrix_[i][j] > 0) d++;
        }
        if (d > max_degree) {
          root = i;
          max_degree = d;
        }
      }

      std::vector<int> q;
      q.push_back(root);
      int i = 0;
      std::vector<int> visited(pat.nnodes, 0);
      while (!q.empty()) {
        int a = q.back();
        q.pop_back();
        if (!visited[a]) {
          vertex_order_[i++] = a;
        }
        visited[a] = 1;
        for (int b = 0; b < pat.nnodes; b++) {
          if (adj_matrix_[a][b] > 0 && !visited[b])
            q.push_back(b);
        }
      }

      for (int i = 0; i < pat.nnodes; i++)
        order_map_[vertex_order_[i]] = i;*/


      

      for (int i = 0; i < pat.nnodes; i++) {
        vertex_order_[i] = i; // idx->vID
        order_map_[vertex_order_[i]] = i; // vID->idx
      }

      for (int i = 0; i < pat.nnodes; i++) {
        int d = 0;
        for (int j = 0; j < pat.nnodes; j++) {
          if (adj_matrix_[i][j] > 0) d++;
        }
        pat.degree[order_map_[i]] = d;
      }
    }

    void _permutation(
      std::vector<std::vector<int>>& all,
      std::vector<int>& a, int l, int r) {
      // Base case
      if (l == r)
        all.push_back(a);
      else {
        // Permutations made
        for (int i = l; i <= r; i++) {
          // Swapping done
          std::swap(a[l], a[i]);
          // Recursion called
          _permutation(all, a, l + 1, r);
          // backtrack
          std::swap(a[l], a[i]);
        }
      }
    }

    void get_set_ops() {

      board.resize(pat.nnodes, std::vector<int>(pat.nnodes, 0));
      board[0][0] = 1;

      for (int i = 1; i < pat.nnodes - 1; i++) {
        int ops = 0;
        for (int j = 0; j <= i; j++) {
          if (adj_matrix_[vertex_order_[i + 1]][vertex_order_[j]]) ops |= (1 << (i - j));
        }
        board[i][0] = ops;
      }

      memset(length, 0, sizeof(length));
      for (int i = 0; i < pat.nnodes; i++) length[i] = 1;

      memset(set_ops, 0, sizeof(set_ops));
      for (int j = 0; j < pat.nnodes - 1; j++) {
        for (int i = pat.nnodes - 2 - j; i >= 0; i--) {
          // 0 means empty slot in board
          if (board[i][j] == 0) continue;

          int op1 = board[i][j] & 1;
          int op2 = (board[i][j] >> 1);

          if (op2 > 0) {
            bool exist = false;
            // k starts from 1 to make sure candidate sets are not used for computing slots 
            //int startk = ((!LABELED && partial[i - 1][0] == 0) ? 0 : 1);
            int startk = 1;
            for (int k = startk; k < length[i - 1]; k++) {
              if (op2 == board[i - 1][k]) {
                exist = true;
                set_ops[i][j] += k;
                set_ops[i][j] += (op1 << 5);
                break;
              }
            }
            if (!exist) {
              set_ops[i][j] += length[i - 1];
              set_ops[i][j] += (op1 << 5);
              board[i - 1][length[i - 1]++] = op2;
            }
          }
          else {
            set_ops[i][j] |= 0x10;
          }
        }
      }
      // mark the end of slot
      for (int i = 0; i < pat.nnodes - 1; i++) {
        set_ops[i][length[i]] |= 0x80;
      }
    }

    void get_partial_order() {

      std::vector<int> p1;
      for (int i = 0; i < pat.nnodes; i++) {
        p1.push_back(i);
      }
      std::vector<std::vector<int>> permute, valid_permute;
      _permutation(permute, p1, 0, pat.nnodes - 1);

      for (auto& pp : permute) {
        std::vector<std::set<int>> adj_tmp(pat.nnodes);
        for (int i = 0; i < pat.nnodes; i++) {
          std::set<int> tp;
          for (int j = 0; j < pat.nnodes; j++) {
            if (adj_matrix_[i][j] == 0) continue;
            tp.insert(pp[j]);
          }
          adj_tmp[pp[i]] = tp;
        }
        bool valid = true;
        for (int i = 0; i < pat.nnodes; i++) {
          bool equal = true;
          int c = 0;
          for (int j = 0; j < pat.nnodes; j++) {
            if (adj_matrix_[i][j] == 1) {
              c++;
              if (adj_tmp[i].find(j) == adj_tmp[i].end()) equal = false;
            }
          }
          if (!equal || c != adj_tmp[i].size()) {
            valid = false;
            break;
          }
        }
        if (valid)
          valid_permute.push_back(pp);
      }

      PatternMultiplicity = valid_permute.size();

      L_adj_matrix_.resize(pat.nnodes, std::vector<int>(pat.nnodes, 0));
      std::set<std::pair<int, int>> L;
      for (int i = 0; i < pat.nnodes; i++) {
        int v = vertex_order_[i];
        std::vector<std::vector<int>> stabilized_aut;
        for (auto& x : valid_permute) {
          if (x[v] == v) {
            stabilized_aut.push_back(x);
          }
          else {
            L_adj_matrix_[order_map_[v]][order_map_[x[v]]] = 1;
          }
        }
        valid_permute = stabilized_aut;
      }

      memset(partial, 0, sizeof(partial));
      for (int level = 1; level < pat.nnodes; level++) {
        for (int j = level - 1; j >= 0; j--) {
          if (L_adj_matrix_[j][level] == 1) {
            partial[level - 1][0] |= (1 << j);
          }
        }
      }
      for (int i=0; i<pat.nnodes; ++i) {
        for (int j=0; j<pat.nnodes; ++j) {
          pat.partial[i][j] = partial[i][j];
        }
      }
      //================== execute condition array ==================
      std::vector<bool> visited(pat.nnodes, false);
      memset(pat.backward_neighbors, 0, sizeof(pat.backward_neighbors));
      memset(pat.num_BN, 0, sizeof(pat.num_BN));

      visited[vertex_order_[0]] = true;

      for(int i = 1; i < pat.nnodes; ++i) { //idx
        int vertex = vertex_order_[i];
        for (int j = 0; j < PAT_SIZE; ++j) { // vID
          if (adj_matrix_[vertex][j] > 0) {
            if (visited[j]) {
              pat.backward_neighbors[i][pat.num_BN[i]++] = order_map_[j]; // for simplicity
            }
          }
        }
        visited[vertex] = true;
      }
      std::cout << "# BNs:\n"; 
      for(int i = 0; i < pat.nnodes; ++i)
      {
        for (int j = 0; j < pat.num_BN[i]; ++j)
        {
          std::cout << pat.backward_neighbors[i][j] << " ";
        } 
        std::cout << "\n";
      }
      std::cout << std::endl;
      // ======================= execute condition array ===============

      memset(pat.condition_order, 0, sizeof(pat.condition_order));
      memset(pat.condition_cnt, 0, sizeof(pat.condition_cnt));
      bool skip;
      for (int i = 0; i < pat.nnodes; ++i) // idx
      {
        int index = i * PAT_SIZE * 2;
        for (int j = 0; j < pat.nnodes; ++j) // idx
        {
          if (i > j)
          {
            skip = false;
            for (int k = 0; k < order_[order_map_[i]].size(); ++k)
            {
              if (order_[order_map_[i]][k].second == j)
              {
                pat.condition_order[index] = order_[order_map_[i]][k].first;
                pat.condition_order[index + 1] = j;
                pat.condition_cnt[i] += 1;
                index += 2;
                skip = true;
                break;
              }
            }
            if (!skip) 
            {
              pat.condition_order[index] = CondOperator::NON_EQUAL;
              pat.condition_order[index + 1] = j;
              pat.condition_cnt[i] += 1;
              index += 2;
            }
          }
        }
      }

      std::cout << "# Conditions:\n"; 
      for(int i = 0; i < pat.nnodes; ++i)
      {
        for (int j = 0; j < pat.condition_cnt[i]; ++j)
        {
          std::cout << pat.condition_order[i * PAT_SIZE * 2 + j * 2] << " " 
                    << pat.condition_order[i * PAT_SIZE * 2 + j * 2 + 1] << " | ";
        } 
        std::cout << "\n";
      }
      std::cout << std::endl;
    }

    int bitidx(bitarray32 a) {
      for (int i = 0; i < 32; i++) {
        if (a & (1 << i)) return i;
      }
      return -1;
    }

    void propagate_partial_order() {
      // propagate partial order of candiate sets to all slots
      for (int i = pat.nnodes - 3; i >= 0; i--) {
        for (int j = 1; j < length[i]; j++) {
          int m = 0;
          // for all slots in the next level, 
          for (int k = 0; k < length[i + 1]; k++) {
            if (set_ops[i + 1][k] & 0x20) {
              // if the slot depends on the current slot and the operation is intersection
              if ((set_ops[i + 1][k] & 0xF) == j) {
                if (partial[i + 1][k] != 0) {
                  // we add the upper bound of that slot to the current slot
                  // the upper bound has to be vertex above level i 
                  m |= (partial[i + 1][k] & (((1 << (i + 1)) - 1)));
                }
                else {
                  m = 0;
                  break;
                }
              }
            }
            else {
              m = 0;
              break;
            }
          }
          partial[i][j] = m;
        }
      }
    }

    void get_labels() {

      memset(slot_labels, 0, sizeof(slot_labels));

      for (int i = 0; i < pat.nnodes; i++) {
        slot_labels[i][0] = (1 << vertex_labels[i + 1]);

        pat.vertex_labels[i] = vertex_labels[i];
      }

      for (int i = pat.nnodes - 3; i >= 0; i--) {
        for (int j = 1; j < length[i]; j++) {

          bitarray32 m = 0;
          //if(j==0) m = pat.partial[i][j];
          // for all slots in the next level, 
          for (int k = 0; k < length[i + 1]; k++) {
            // if the slot depends on the current slot and the operation is intersection
            if ((set_ops[i + 1][k] & 0xF) == j) {
              // we add the upper bound of that slot to the current slot
              // the upper bound has to be vertex above level i 
              m |= slot_labels[i + 1][k];
            }
          }
          slot_labels[i][j] = m;
        }
      }
    }

    void convert2oned() {

      int onedidx[PAT_SIZE][PAT_SIZE];
      memset(onedidx, 0, sizeof(onedidx));

      int count = 1;
      pat.rowptr[0] = 0;
      pat.rowptr[1] = 1;
      // this is used for filtering the edges in job queue
      pat.partial[0] = partial[0][0];
      for (int i = 1; i < pat.nnodes - 1; i++) {
        for (int j = 0; j < PAT_SIZE; j++) {
          if (set_ops[i][j] < 0) break;
          onedidx[i][j] = count;
          pat.slot_labels[count] = slot_labels[i][j];
          pat.partial[count] = partial[i][j];
          int idx = 0;
          if (i > 1) idx = onedidx[i - 1][(set_ops[i][j] & 0x0F)];
          assert(idx < 31);
          pat.set_ops[count] = ((set_ops[i][j] & 0x30) << 1) + idx;
          count++;
        }
        pat.rowptr[i + 1] = count;
      }
      //std::cout << "total number of slots: " << count << std::endl;
      assert(count <= MAX_SLOT_NUM);
    }

    bliss::Graph* GetBlissGraph()
    {
      bliss::Graph* bg = new bliss::Graph(pat.nnodes);
      for (int i = 0; i < pat.nnodes; i++)
      {
        for (int j = 0; j < pat.nnodes; ++j) {
          if (adj_matrix_[i][j] > 0)
          {
            bg->add_edge(i, j);
          }
        }
      }
      return bg;
    }

    std::string OrderToString(const std::vector<int>& p) {
      std::string res;
      for (auto v : p)
        res += std::to_string(v);
      return res;
    }

    void CompleteAutomorphisms(std::vector<std::vector<int>>& perm_group) 
    {
        // multiplying std::vector<uint32_t>s is just function composition: (p1*p2)[i] = p1[p2[i]]
        std::vector<std::vector<int>> products;
        // for filtering duplicates
        std::unordered_set<std::string> dups;
        for (auto f : perm_group)
          dups.insert(OrderToString(f));

        for (auto k = perm_group.begin(); k != perm_group.end(); k++) 
        {
          for (auto l = perm_group.begin(); l != perm_group.end(); l++) 
          {
            std::vector<int> p1 = *k;
            std::vector<int> p2 = *l;

            std::vector<int> product;
            product.resize(p1.size());
            for (int i = 0; i < product.size(); i++)
                product[i] = p1[p2[i]];

            // don't count duplicates
            if (dups.count(OrderToString(product)) == 0) 
            {
                dups.insert(OrderToString(product));
                products.push_back(product);
            }
          }
        }

        for (auto p : products)
          perm_group.push_back(p);
    }

    std::vector<std::vector<int>> GetAutomorphisms(bliss::Graph* bg)
    {
        std::vector<std::vector<int>> result;
        bliss::Stats stats;
        bg->find_automorphisms(
            stats,
            [](void* param, const unsigned int size, const unsigned int* aut) {
                std::vector<int> result_aut;
                for (int i = 0; i < size; i++)
                    result_aut.push_back(aut[i]);
                ((std::vector<std::vector<int>>*)param)->push_back(result_aut);
            },
            &result);

        int counter = 0;
        int lastSize = 0;
        while (result.size() != lastSize) 
        {
            lastSize = result.size();
            CompleteAutomorphisms(result);
            counter++;
            if (counter > 100)
                break;
        }

        return result;
    }

    std::map<int, std::set<int>> GetAEquivalenceClasses(const std::vector<std::vector<int>>& aut) 
    {
      std::map<int, std::set<int>> eclasses;
      for (int i = 0; i < pat.nnodes; i++) 
      {
          std::set<int> eclass;
          for (auto&& perm : aut)
            eclass.insert(perm[i]);
          int rep = *std::min_element(eclass.cbegin(), eclass.cend());
          eclasses[rep].insert(eclass.cbegin(), eclass.cend());
      }
      return eclasses;
    }

    std::vector<std::pair<int, int>> GetConditions(bliss::Graph* bg)
    {
        std::vector<std::vector<int>> aut = GetAutomorphisms(bg);
        std::map<int, std::set<int>> eclasses = GetAEquivalenceClasses(aut);

        std::vector<std::pair<int, int>> result;
        auto eclass_it = std::find_if(eclasses.cbegin(), eclasses.cend(), [](auto&& e) { return e.second.size() > 1; });
        while (eclass_it != eclasses.cend() && eclass_it->second.size() > 1) 
        {
            const auto& eclass = eclass_it->second;
            int n0 = *eclass.cbegin();

            for (auto&& perm : aut)
            {
                int min = *std::min_element(std::next(eclass.cbegin()), eclass.cend(), [perm](int n, int m) { return perm[n] < perm[m]; });
                result.emplace_back(n0, min);
            }
            aut.erase(std::remove_if(aut.begin(), aut.end(), [n0](auto&& perm) { return perm[n0] != n0; }), aut.end());

            eclasses = GetAEquivalenceClasses(aut);
            eclass_it = std::find_if(eclasses.cbegin(), eclasses.cend(), [](auto&& e) { return e.second.size() > 1; });
        }

        // remove duplicate conditions
        std::sort(result.begin(), result.end());
        result.erase(std::unique(result.begin(), result.end()), result.end());

        return result;
    }

    void SetConditions(const std::vector<std::pair<int, int>>& conditions) 
    {
        order_.resize(pat.nnodes);
        for (int i = 0; i < conditions.size(); i++) 
        {
            int first = conditions[i].first;
            int second = conditions[i].second;
            order_[first].push_back(std::make_pair(LESS_THAN, second));
            order_[second].push_back(std::make_pair(LARGER_THAN, first));
        }
    }
  };
}