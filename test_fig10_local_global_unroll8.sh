#!/bin/bash
graph_path=./data/bin_graph/
pattern_path=./data/pattern/
patterns=(9.g  10.g  11.g  12.g  13.g  14.g  15.g  16.g)

echo "Local Steal+Global Steal+Unroll8"
for pattern in ${patterns[@]}
do    
    ./bin/fig_local_global_unroll.exe ${1}/snap.txt ${pattern_path}/${pattern}
done
