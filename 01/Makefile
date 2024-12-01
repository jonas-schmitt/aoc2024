# Compiler settings
CXX = g++
NVCC = nvcc
CXXFLAGS = -Wall -Werror -std=c++23 -O3
NVCCFLAGS = -O3

# Target executable names
TARGETS = ex1 ex2 ex1_cuda ex2_cuda

all: $(TARGETS)

# Build rules for C++ executables
ex1: ex1.cpp
	$(CXX) $(CXXFLAGS) ex1.cpp -o ex1

ex2: ex2.cpp
	$(CXX) $(CXXFLAGS) ex2.cpp -o ex2

# Build rules for CUDA executables
ex1_cuda: ex1.cu
	$(NVCC) $(NVCCFLAGS) ex1.cu -o ex1_cuda

ex2_cuda: ex2.cu
	$(NVCC) $(NVCCFLAGS) ex2.cu -o ex2_cuda

# Clean rule
clean:
	rm -f $(TARGETS)

.PHONY: all clean