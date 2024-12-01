#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <algorithm>
#include <numeric>

int main() {
    int a, b;
    std::vector<int> left, right;
    std::unordered_map<int,int> counts;
    while (std::cin >> a >> b) {
        counts[a] = 0;
        left.push_back(a);
        right.push_back(b);
    }
    std::for_each(right.begin(), right.end(), [&counts](auto i) {
        if(counts.contains(i)) {
            ++counts[i];
        }
    });
    auto res = std::transform_reduce(
        left.begin(), left.end(), 0, std::plus<>(),
        [&counts](auto a) {
            return a*counts[a];
        }
    );


    std::cout << res << std::endl; 
    return 0;
}