#include <ranges>
#include <string_view>

#include <parser.hpp>

auto main() -> int {
    unsigned long fsize {};

    char current_working_directory[MAX_PATH] {};
    ::GetCurrentDirectoryA(MAX_PATH, current_working_directory);
    ::puts(current_working_directory);

    auto                       beans { ::open(L"dry_beans.csv", &fsize) };
    const std::string_view     beans_view { beans };
    constexpr std::string_view CRLF { "\r\n" };
    constexpr std::string_view delimiter { "," };

    const auto lines = std::ranges::split_view(beans_view, CRLF);
    for (const auto line : lines);
    return EXIT_SUCCESS;
}