#pragma once

#include <stdlib.h> 
#include <string.h>
#include <stdio.h>
#include "config.h"

namespace STMatch {

   typedef struct {
    graph_node_t iter[PAT_SIZE];
    graph_node_t uiter[PAT_SIZE];
    graph_node_t slot_size[MAX_SLOT_NUM];
    graph_node_t (*slot_storage)[GRAPH_DEGREE];
    pattern_node_t level;
    bool stealed_task = false;
  } CallStack;

/*
  void init() {
    memset(path, 0, sizeof(path));
    memset(iter, 0, sizeof(iter));
    memset(slot_size, 0, sizeof(slot_size));
  }
  */
}