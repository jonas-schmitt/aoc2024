# Compiler settings
CXX = g++
CXXFLAGS = -Wall -Werror -std=c++23 -O3

# Target executable names
TARGETS = ex1 ex2

all: $(TARGETS)

# Build rules for C++ executables
ex1: ex1.cpp
	$(CXX) $(CXXFLAGS) ex1.cpp -o ex1

ex2: ex2.cpp
	$(CXX) $(CXXFLAGS) ex2.cpp -o ex2

# Clean rule
clean:
	rm -f $(TARGETS)

.PHONY: all clean