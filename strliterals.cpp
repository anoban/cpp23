// can we differentiate const char (&)[] from const char* in C++???

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <new>
#include <source_location>
#include <unix>

template<size_t length> static inline void print(
    _In_ const char (&string)[length], _In_opt_ const std::source_location& sloc = std::source_location::current()
) noexcept {
    ::printf_s("%s invoked @ line %u\n", __PRETTY_FUNCTION__, sloc.line()); // NOLINT(cppcoreguidelines-pro-type-vararg)
    ::puts(string);
}

// another overload of print
static inline void print(
    _In_ const char* const string, _In_opt_ const std::source_location& sloc = std::source_location::current()
) noexcept {
    ::printf_s("%s invoked @ line %u\n", __PRETTY_FUNCTION__, sloc.line()); // NOLINT(cppcoreguidelines-pro-type-vararg)
    ::puts(string);
}

auto main() -> int {
    const char* dido { "Thank you!" };
    ::print(dido);
    ::print<>("Hi there Anoban!"); // to invoke the templatized overload we need to explicitly instantiate the template
    // otherwise it would call the non-template function

    std::ifstream file { R"(./romeojuliet.txt)", std::ios::in | std::ios::ate };
    if (!file.is_open()) return EXIT_FAILURE;
    auto nbytes { file.tellg() - file.seekg(std::ios::beg).tellg() };

    const auto fsize { std::filesystem::file_size(R"(./romeojuliet.txt)") };
    printf_s("std::ifstream %lld, std::filesystem %llu \n", nbytes, fsize); // NOLINT(cppcoreguidelines-pro-type-vararg)

    auto str = std::make_unique_for_overwrite<char[]>(nbytes);
    file.read(str.get(), nbytes);
    // ::puts(str);

    ::print(str.get());

    return EXIT_SUCCESS;
}
