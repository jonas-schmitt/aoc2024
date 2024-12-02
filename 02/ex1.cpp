#include <iostream>
#include <sstream>
#include <vector>
#include <string>

int main() {
    std::string line;
    size_t count = 0;
    std::vector<int> numbers;
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
        int const diff_0 = numbers[1] - numbers[0];
        if(diff_0 == 0) {
            continue;
        }
        ++count;
        for(size_t i = 0; i < numbers.size()-1; i+=1) {
            int const diff_i = numbers[i+1] - numbers[i];
            int const tmp = std::abs(diff_i);
            if(diff_0 * diff_i < 0 || tmp < 1 || tmp > 3) {
                --count;
                break;
            }
        } 
    }
    std::cout << count << std::endl;
    
    return 0;
}