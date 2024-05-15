DEBUG = 

OPTIONS = -arch=sm_80 -gencode=arch=compute_80,code=sm_80 -Xptxas -v -maxrregcount 64 -Lbliss -lbliss --expt-relaxed-constexpr
GPU_MATCH = src/gpu_match.cu 

define compile_cu_test
	nvcc -std=c++17 $(DEBUG) $(OPTIONS) $(1) cu_test.cu -DLABELED=$(3) -I. -I./src/Ouroboros/include -o $(2)
endef

define compile_gpu_match
	nvcc -std=c++17 $(DEBUG) $(OPTIONS) -dc -I. -I./src/Ouroboros/include $(1) -o $(2)
endef

.PHONY:all

all:bin/table_edge_ulb.exe bin/table_edge_lb.exe

bin/table_edge_ulb.exe:
	$(call compile_cu_test,$<,$@,false)

bin/table_edge_lb.exe:
	$(call compile_cu_test,$<,$@,true)

.PHONY:clean
clean:
	rm -f bin/*.o bin/*.exe