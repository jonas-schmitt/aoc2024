#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <numeric>

int main(int argc, char **argv) {
    int a, b;
    std::vector<int> left, right;
    while (std::cin >> a >> b) {
        left.push_back(a);
        right.push_back(b);
    }
    std::sort(left.begin(), left.end());
    std::sort(right.begin(), right.end());
    std::vector<int> dist;
    dist.resize(left.size());
    auto sum = std::transform_reduce(
        left.begin(), left.end(), 
        right.begin(),              
        0,                          
        std::plus<>(),              
        [](int a, int b) {          
            return std::abs(a - b);
        });
    std::cout << sum << std::endl;
    
}