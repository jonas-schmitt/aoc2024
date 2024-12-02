#include <iostream>
#include <sstream>
#include <vector>
#include <string>

std::pair<bool, size_t> is_safe(const std::vector<int>& numbers)
{
    int const diff_0 = numbers[1] - numbers[0];
    if (diff_0 == 0)
    {
        return {false, 0};  // Using brace initialization
    }
    
    for (size_t i = 0; i < numbers.size() - 1; i += 1)
    {
        int const diff_i = numbers[i + 1] - numbers[i];
        int const tmp = std::abs(diff_i);
        if (diff_0 * diff_i < 0 || tmp < 1 || tmp > 3)
        {
            return {false, i};
        }
    }
    return {true, 0};
}

// Use a reference to avoid copying the result vector multiple times
void remove_element(const std::vector<int>& vec, size_t pos, std::vector<int>& result) {
    result.clear();
    result.reserve(vec.size() - 1);
    for (size_t i = 0; i < vec.size(); ++i) {
        if (i != pos) {
            result.push_back(vec[i]);
        }
    }
}

int main() {
    std::string line;
    size_t count = 0;
    std::vector<int> numbers;
    std::vector<int> temp;
    
    while (std::getline(std::cin, line)) {
        std::istringstream iss(line);
        int number;
        numbers.clear();
        while (iss >> number) {
            numbers.push_back(number);
        }
        
        if(numbers.size() <= 1) {
            continue;
        }
        
        auto res = is_safe(numbers);
        if (res.first) {
            ++count;
            continue;
        }
        
        bool can_fix = false;
        if (res.second > 0) {
            remove_element(numbers, res.second - 1, temp);
            if (is_safe(temp).first) {
                can_fix = true;
            }
        }
        
        if (!can_fix) {
            remove_element(numbers, res.second, temp);
            if (is_safe(temp).first) {
                can_fix = true;
            }
        }
        
        if (!can_fix && res.second + 1 < numbers.size()) {
            remove_element(numbers, res.second + 1, temp);
            if (is_safe(temp).first) {
                can_fix = true;
            }
        }
        
        if (can_fix) {
            ++count;
        }
    }
    
    std::cout << count << std::endl;
    return 0;
}