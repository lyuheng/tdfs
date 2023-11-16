DEBUG =

OPTIONS = -arch=sm_80 -gencode=arch=compute_80,code=sm_80 -Xptxas -v -maxrregcount 64 -Lbliss -lbliss
GPU_MATCH = src/gpu_match.cu 

define compile_cu_test
	nvcc -std=c++17 $(DEBUG) $(OPTIONS) $(1) cu_test.cu -o $(2) -DLABELED=$(3)
endef

define compile_gpu_match
	nvcc -std=c++17 $(DEBUG) $(OPTIONS) -dc -I. $(1) -o $(2) -DLABELED=$(3)
endef

.PHONY:all
all:bin/lb.out bin/ulb.out

bin/ulb.out: bin/ulb.o;
	$(call compile_cu_test,$<,$@,false)
bin/lb.out: bin/lb.o;
	$(call compile_cu_test,$<,$@,true)
bin/%.o:
	$(MKDIR_P) $(dir $@)
	$(call compile_gpu_match,src/gpu_match.cu,bin/ulb.o,false)
	$(call compile_gpu_match,src/gpu_match.cu,bin/lb.o,true)

.PHONY:clean
clean:
	rm -f bin/*.o bin/*.out

MKDIR_P = mkdir -p