#include <parser.hpp>

template<typename T>
[[nodiscard]] static std::enable_if<std::is_floating_point<T>::value, std::vector<::record<T>>>::type parse_beans_records(
    _In_ const std::string& text, _In_ const bool& has_header
) noexcept {
    const auto nlines { std::ranges::count(text, '\n') }; // 13,612
    assert(nlines == 13'612);

    record<T>                placeholder {};
    std::vector<::record<T>> collection {};
    collection.reserve(nlines); // space for 1 extra record is there since we will not parse the header

    auto header_end { has_header ? text.find('\n') : 0LLU };
    // 28395,610.291,208.178116708527,173.888747041636,1.19719142411602,0.549812187138347,28715,190.141097274511,0.763922518159806,0.988855998607,0.958027126250128,0.913357754795763,0.00733150613518321,0.00314728916733569,0.834222388245556,0.998723889013168,SEKER

    auto next_line_end { text.find('\n', header_end + 1) };

    return collection;
}

auto main() -> int {
    static char current_working_directory[MAX_PATH] {};
    ::GetCurrentDirectoryA(MAX_PATH, current_working_directory);
    ::puts(current_working_directory);

    unsigned long fsize {};
    auto          beans { ::open(LR"(dry_beans.csv)", &fsize) };
    ::puts(beans.c_str());

    return EXIT_SUCCESS;
}
