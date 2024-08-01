#include <parser.hpp>

auto main() -> int {
    static char current_working_directory[MAX_PATH] {};
    ::GetCurrentDirectoryA(MAX_PATH, current_working_directory);
    ::puts(current_working_directory);

    unsigned long          fsize {};
    auto                   beans { ::open(LR"(dry_beans.csv)", &fsize) };
    const std::string_view beans_view { beans };

    const auto nlines { std::ranges::count(beans, '\n') }; // 13,612
    ::printf_s("%lld lines\n", nlines);

    constexpr auto CRLF { '\n' };
    constexpr auto COMMA { ',' };

    const auto lines = std::ranges::views::split(beans_view, CRLF);
    for (const auto line : lines) ::puts(line);

    return EXIT_SUCCESS;
}