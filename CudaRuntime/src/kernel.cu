#include <parser.hpp>

auto main() -> int {
    unsigned long fsize {};

    char cwd[MAX_PATH] {};
    ::GetCurrentDirectoryA(MAX_PATH, cwd);
    ::puts(cwd);

    ::puts(::open(L".\dry_beans.csv", &fsize).c_str());
    return EXIT_SUCCESS;
}