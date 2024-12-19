#include <fstream>
#include <iostream>

int main() {
    std::ifstream cpp { LR"(./cppctors.png)", std::ios::binary | std::ios::in };
    if (!cpp.is_open()) return EXIT_FAILURE;

    std::cout << cpp.tellg() << '\n';

    cpp.seekg(std::ios::end);
    std::cout << cpp.tellg() << '\n';

    cpp.seekg(std::ios::beg);
    std::cout << cpp.tellg() << '\n';

    while(cpp.get())

    return EXIT_SUCCESS;
}
