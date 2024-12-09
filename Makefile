# Combined Makefile for all subfolders
CXX = g++

# Check if NVCC is available
NVCC_CHECK := $(shell which nvcc 2>/dev/null)
ifdef NVCC_CHECK
    NVCC = nvcc
    CUDA_AVAILABLE = 1
else
    CUDA_AVAILABLE = 0
endif

# Debug flag (can be set via DEBUG=1)
DEBUG ?= 0

# Common C++ flags
CXXFLAGS_COMMON = -Wall -Werror -std=c++23

# Add debug/release specific flags
ifeq ($(DEBUG), 1)
    CXXFLAGS_COMMON += -g -O0 -DDEBUG
    NVCCFLAGS = -g -G -O0
    BUILD_TYPE = debug
else
    CXXFLAGS_COMMON += -O3 -DNDEBUG
    NVCCFLAGS = -O3
    BUILD_TYPE = release
endif

# Create build directories for debug/release
BUILD_DIR = build/$(BUILD_TYPE)

# Folder-specific flags
CXXFLAGS_01 = $(CXXFLAGS_COMMON)
CXXFLAGS_02 = $(CXXFLAGS_COMMON)
CXXFLAGS_03 = $(CXXFLAGS_COMMON)
CXXFLAGS_04 = $(CXXFLAGS_COMMON)

# Define targets for each folder with build directory
TARGETS_01_CPP = $(BUILD_DIR)/01/ex1 $(BUILD_DIR)/01/ex2
TARGETS_01_CUDA = $(BUILD_DIR)/01/ex1_cuda $(BUILD_DIR)/01/ex2_cuda
TARGETS_02 = $(BUILD_DIR)/02/ex1 $(BUILD_DIR)/02/ex2
TARGETS_03 = $(BUILD_DIR)/03/ex1 $(BUILD_DIR)/03/ex2
TARGETS_04 = $(BUILD_DIR)/04/ex1 $(BUILD_DIR)/04/ex2

# Conditionally include CUDA targets
ifeq ($(CUDA_AVAILABLE), 1)
    TARGETS_01 = $(TARGETS_01_CPP) $(TARGETS_01_CUDA)
else
    TARGETS_01 = $(TARGETS_01_CPP)
endif

# Combine all targets
ALL_TARGETS = $(TARGETS_01) $(TARGETS_02) $(TARGETS_03) $(TARGETS_04)

# Default target
all: create_dirs $(ALL_TARGETS)

# Create necessary build directories
create_dirs:
	@mkdir -p $(BUILD_DIR)/01
	@mkdir -p $(BUILD_DIR)/02
	@mkdir -p $(BUILD_DIR)/03
	@mkdir -p $(BUILD_DIR)/04

# Rules for folder 01 (includes CUDA)
$(BUILD_DIR)/01/ex1: 01/ex1.cpp
	$(CXX) $(CXXFLAGS_01) $< -o $@

$(BUILD_DIR)/01/ex2: 01/ex2.cpp
	$(CXX) $(CXXFLAGS_01) $< -o $@

# CUDA rules only if available
ifeq ($(CUDA_AVAILABLE), 1)
$(BUILD_DIR)/01/ex1_cuda: 01/ex1.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(BUILD_DIR)/01/ex2_cuda: 01/ex2.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@
endif

# Rules for folder 02
$(BUILD_DIR)/02/ex1: 02/ex1.cpp
	$(CXX) $(CXXFLAGS_02) $< -o $@

$(BUILD_DIR)/02/ex2: 02/ex2.cpp
	$(CXX) $(CXXFLAGS_02) $< -o $@

# Rules for folder 03
$(BUILD_DIR)/03/ex1: 03/ex1.cpp
	$(CXX) $(CXXFLAGS_03) $< -o $@

$(BUILD_DIR)/03/ex2: 03/ex2.cpp
	$(CXX) $(CXXFLAGS_03) $< -o $@

# Rules for folder 04
$(BUILD_DIR)/04/ex1: 04/ex1.cpp
	$(CXX) $(CXXFLAGS_04) $< -o $@

$(BUILD_DIR)/04/ex2: 04/ex2.cpp
	$(CXX) $(CXXFLAGS_04) $< -o $@

# Clean all targets
clean:
	rm -rf build

# Show current build configuration
info:
	@echo "Build type: $(BUILD_TYPE)"
	@echo "C++ flags: $(CXXFLAGS_COMMON)"
	@echo "CUDA available: $(CUDA_AVAILABLE)"
ifeq ($(CUDA_AVAILABLE), 1)
	@echo "CUDA flags: $(NVCCFLAGS)"
endif

.PHONY: all clean info create_dirs