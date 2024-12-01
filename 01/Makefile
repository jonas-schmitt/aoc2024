# Compiler settings
CXX = g++
CXXFLAGS = -Wall -Werror -std=c++23 -O3

# Target executable name
all: ex1 ex2
TARGET1 = ex1
TARGET2 = ex2

# Build rule for final executable
$(TARGET1): ex1.cpp
	$(CXX) $(CXXFLAGS) ex1.cpp -o $(TARGET1)

$(TARGET2): ex2.cpp
	$(CXX) $(CXXFLAGS) ex2.cpp -o $(TARGET2)

# Clean rule
clean:
	rm -f $(TARGET)