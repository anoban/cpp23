#include <fstream>
#include <iostream>

int main() {
    try {
        std::ifstream cpp { LR"(./cppctors.png)", std::ios::binary | std::ios::in };
        if (!cpp.is_open()) return EXIT_FAILURE;

        std::cout << cpp.tellg() << '\n';

        cpp.seekg(std::ios::end);
        std::cout << cpp.tellg() << '\n';

        cpp.seekg(std::ios::beg);
        std::cout << cpp.tellg() << '\n';

        while (true)
            if (!cpp.get()) {
                std::cout << cpp.tellg() << '\n';
                return EXIT_FAILURE;
            }
    } catch (const std::exception& excpt) {
        std::cout << excpt.what() << '\n';
        return EXIT_FAILURE;
    }
}
