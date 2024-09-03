#include <parser.cuh>

static constexpr size_t N_RECORDS { 13'612 };

template<std::floating_point T> [[nodiscard]] static constexpr record<T> parse_line(_In_ const std::string_view& line) noexcept {
    // a typical row will be in the format of,
    // 28395,610.291,208.178116708527,173.888747041636,1.19719142411602,0.549812187138347,28715,190.141097274511,0.763922518159806,0.988855998607,0.958027126250128,0.913357754795763,0.00733150613518321,0.00314728916733569,0.834222388245556,0.998723889013168,SEKER
    record<T>         temporary {};
    const char* const cstart { line.data() };
    const char*       begin { line.data() };

    size_t caret = line.find(',', 0); // the first comma
    std::from_chars(begin, cstart + caret /* this delimiter is exclusive */, temporary.area);
    begin = cstart + caret + 1; // char next to the first comma

    caret = line.find(',', caret + 1); // the second comma
    std::from_chars(begin, /* char next to the comma */ cstart + caret, temporary.perimeter);
    begin = cstart + caret + 1;

    caret = line.find(',', caret + 1);
    std::from_chars(begin, cstart + caret, temporary.major_axis_length);
    begin = cstart + caret + 1;

    caret = line.find(',', caret + 1);
    std::from_chars(begin, cstart + caret, temporary.minor_axis_length);
    begin = cstart + caret + 1;

    caret = line.find(',', caret + 1);
    std::from_chars(begin, cstart + caret, temporary.aspect_ratio);
    begin = cstart + caret + 1;

    caret = line.find(',', caret + 1);
    std::from_chars(begin, cstart + caret, temporary.eccentricity);
    begin = cstart + caret + 1;

    caret = line.find(',', caret + 1);
    std::from_chars(begin, cstart + caret, temporary.convex_area);
    begin = cstart + caret + 1;

    caret = line.find(',', caret + 1);
    std::from_chars(begin, cstart + caret, temporary.equiv_diameter);
    begin = cstart + caret + 1;

    caret = line.find(',', caret + 1);
    std::from_chars(begin, cstart + caret, temporary.extent);
    begin = cstart + caret + 1;

    caret = line.find(',', caret + 1);
    std::from_chars(begin, cstart + caret, temporary.roundness);
    begin = cstart + caret + 1;

    caret = line.find(',', caret + 1);
    std::from_chars(begin, cstart + caret, temporary.compactness);
    begin = cstart + caret + 1;

    caret = line.find(',', caret + 1);
    std::from_chars(begin, cstart + caret, temporary.extent);
    begin = cstart + caret + 1;

    caret = line.find(',', caret + 1);
    std::from_chars(begin, cstart + caret, temporary.shape_factor_1);
    begin = cstart + caret + 1;

    caret = line.find(',', caret + 1);
    std::from_chars(begin, cstart + caret, temporary.shape_factor_2);
    begin = cstart + caret + 1;

    caret = line.find(',', caret + 1);
    std::from_chars(begin, cstart + caret, temporary.shape_factor_3);
    begin = cstart + caret + 1;

    caret = line.find(',', caret + 1);
    std::from_chars(begin, cstart + caret, temporary.shape_factor_4);
    begin = cstart + caret + 1;

    // handle the string literal @ the end

    return temporary;
}

template<typename T>
[[nodiscard]] static std::enable_if<std::is_floating_point<T>::value, std::vector<::record<T>>>::type parse_beans_csv(
    _In_ const std::string& csv, _In_ const bool& has_header
) noexcept {
    const auto nlines { std::ranges::count(csv, '\n') }; // 13,612
    assert(nlines == N_RECORDS);

    std::vector<::record<T>> records {};
    records.reserve(nlines); // space for 1 extra record is there since we will not parse the header

    size_t line_begin { has_header ? csv.find('\n', 0) : 0 };
    size_t line_end {};

    while ((line_end = csv.find('\n', line_begin + 1)) != std::string::npos) {
        // create a temporary, given delimiters
        records.push_back(::parse_line<T>(std::string_view { csv.data() + line_begin, csv.data() + line_end }));
        line_begin = line_end;
    }

    return records;
}

auto main() -> int {
    static char current_working_directory[MAX_PATH] {};
    ::GetCurrentDirectoryA(MAX_PATH, current_working_directory);
    ::puts(current_working_directory);

    unsigned long fsize {};
    std::string   beans { ::open(LR"(dry_beans.csv)", &fsize) };

    const auto rows { ::parse_beans_csv<float>(beans, true) };

    std::cout << std::setprecision(15);
    for (const auto& row : rows) std::cout << row;

    return EXIT_SUCCESS;
}
