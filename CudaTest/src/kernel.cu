#include <parser.cuh>

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

template<std::floating_point T> static constexpr record<T> parse_line(const std::string_view& line) noexcept {
    // a typical row will be in the format of,
    // 28395,610.291,208.178116708527,173.888747041636,1.19719142411602,0.549812187138347,28715,190.141097274511,0.763922518159806,0.988855998607,0.958027126250128,0.913357754795763,0.00733150613518321,0.00314728916733569,0.834222388245556,0.998723889013168,SEKER
    record<T> temporary {};
    size_t    caret {};
    std::from_chars<unsigned>(line.substr(0, caret = line.find(',') - 1 /* char before the comma */), temporary.area);
    std::from_chars<T>(
        line.substr(caret += 2 /* char next to the comma */, caret = line.find(',', caret) - 1 /* char before the next coma */),
        temporary.perimeter
    );
    std::from_chars<T>(line.substr(caret += 2, caret = line.find(',', caret) - 1), temporary.major_axis_length);
    std::from_chars<T>(line.substr(caret += 2, caret = line.find(',', caret) - 1), temporary.minor_axis_length);
    std::from_chars<T>(line.substr(caret += 2, caret = line.find(',', caret) - 1), temporary.aspect_ratio);
    std::from_chars<T>(line.substr(caret += 2, caret = line.find(',', caret) - 1), temporary.eccentricity);
    std::from_chars<T>(line.substr(caret += 2, caret = line.find(',', caret) - 1), temporary.convex_area);
    std::from_chars<T>(line.substr(caret += 2, caret = line.find(',', caret) - 1), temporary.equiv_diameter);
    std::from_chars<T>(line.substr(caret += 2, caret = line.find(',', caret) - 1), temporary.extent);
    std::from_chars<T>(line.substr(caret += 2, caret = line.find(',', caret) - 1), temporary.roundness);
    std::from_chars<T>(line.substr(caret += 2, caret = line.find(',', caret) - 1), temporary.compactness);
    std::from_chars<T>(line.substr(caret += 2, caret = line.find(',', caret) - 1), temporary.extent);
    std::from_chars<T>(line.substr(caret += 2, caret = line.find(',', caret) - 1), temporary.shape_factor_1);
    std::from_chars<T>(line.substr(caret += 2, caret = line.find(',', caret) - 1), temporary.shape_factor_2);
    std::from_chars<T>(line.substr(caret += 2, caret = line.find(',', caret) - 1), temporary.shape_factor_3);
    std::from_chars<T>(line.substr(caret += 2, caret = line.find(',', caret) - 1), temporary.shape_factor_4);
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
