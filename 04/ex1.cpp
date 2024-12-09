#include <iostream>
#include <string>
#include <vector>
#include <fstream>

inline bool is_valid_move(size_t from_pos, size_t to_pos, size_t cols, size_t total_size) {
    if (to_pos >= total_size) return false;
    if (to_pos == from_pos + 1 && (from_pos % cols) == cols - 1) return false;
    if (to_pos == from_pos - 1 && (from_pos % cols) == 0) return false;
    return true;
}

int eval_pos(std::string const& text, size_t const text_pos, char const target) {
    return text_pos < text.size() && text[text_pos] == target ? 1 : 0;
}

size_t count_xmas(std::string const& text, size_t const text_pos, size_t const cols, 
                  std::string const& targets, size_t const target_pos, size_t const old_position) {
    size_t count = 0;
    std::vector<size_t> next_positions;

    if(old_position == text_pos) {
        for(int offset : {1, -1, static_cast<int>(cols), static_cast<int>(cols)+1, 
                         static_cast<int>(cols)-1, -static_cast<int>(cols), 
                         -static_cast<int>(cols)+1, -static_cast<int>(cols)-1}) {
            auto next_pos = static_cast<int>(text_pos) + offset;
            if (next_pos >= 0 && is_valid_move(text_pos, next_pos, cols, text.size())) {
                next_positions.push_back(next_pos);
            }
        }
    } else {
        auto next_pos = text_pos + (text_pos - old_position);
        if (is_valid_move(text_pos, next_pos, cols, text.size())) {
            next_positions.push_back(next_pos);
        }
    }

    int res = eval_pos(text, text_pos, targets[target_pos]);
    if(target_pos == targets.size()-1 || res == 0) {
        return res;
    }
    
    for(auto p : next_positions) {
        count += count_xmas(text, p, cols, targets, target_pos+1, text_pos);
    }
    return count;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <filename>" << std::endl;
        return 1;
    }

    std::ifstream file(argv[1]);
    if (!file) {
        std::cerr << "Could not open file: " << argv[1] << std::endl;
        return 1;
    }

    std::string buf;
    size_t count = 0;
    std::string text;
    size_t cols = 0;

    while (std::getline(file, buf)) {
        if(cols == 0) cols = buf.size();
        text.append(buf);
    }

    std::string targets = "XMAS";
    for(size_t i = 0; i < text.size(); ++i) {
        if(text[i] == targets.front()) {
            count += count_xmas(text, i, cols, targets, 0, i);
        }
    }

    std::cout << count << std::endl;
    return 0;
}