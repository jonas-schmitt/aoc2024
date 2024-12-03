#include <iostream>
#include <string>
#include <regex>

int main() {
    std::string text;
    int res = 0;
    // Capture groups () will extract just the numbers
    std::regex pattern(R"(mul\(([1-9]\d?\d?|0),([1-9]\d?\d?|0)\))");
    
    while (std::getline(std::cin, text)) {
        std::smatch match;
        while (std::regex_search(text, match, pattern)) {
            int num1 = std::stoi(match[1]);
            int num2 = std::stoi(match[2]);
            text = match.suffix();
            res += num1 * num2;
        }
    }
    std::cout << res << std::endl;
    return 0;
}