#include <iostream>
#include <string>
#include <regex>

int main() {
    std::string text;
    int res = 0;
    std::regex pattern(R"(mul\(([1-9]\d?\d?|0),([1-9]\d?\d?|0)\)|don't\(\)|do\(\))");
    
    bool disabled = false;
    while (std::getline(std::cin, text)) {
        std::smatch match;
        
        while (std::regex_search(text, match, pattern)) {
            if(match[0] == "do()") {
                disabled = false;
            }
            else if(match[0] == "don't()") {
                disabled = true;
            }
            else if(!disabled) {
                int num1 = std::stoi(match[1]);
                int num2 = std::stoi(match[2]);
                res += num1 * num2;
            }
            
            text = match.suffix();
        }
    }
    std::cout << res << std::endl;
    return 0;
}