#include <charconv>
#include <iostream>
#include <string_view>

template<typename T> class record final {
    public:
        using value_type = T;
        unsigned area;
        T        perimeter;
        T        major_axis_length;
        T        minor_axis_length;
        T        aspect_ratio;
        T        eccentricity;
        T        convex_area;
        T        equiv_diameter;
        T        extent;
        T        solidity;
        T        roundness;
        T        compactness;
        T        shape_factor_1;
        T        shape_factor_2;
        T        shape_factor_3;
        T        shape_factor_4;
        char     variety[10]; // max is 9 so :)

        template<typename char_t>
        friend std::basic_ostream<char_t>& operator<<(_Inout_ std::basic_ostream<char_t>& ostream, _In_ const record& rcrd) noexcept {
            ostream << rcrd.area << char_t(' ') << rcrd.perimeter << char_t(' ') << rcrd.major_axis_length << char_t(' ')
                    << rcrd.minor_axis_length << char_t(' ') << rcrd.aspect_ratio << char_t(' ') << rcrd.eccentricity << char_t(' ')
                    << rcrd.convex_area << char_t(' ') << rcrd.equiv_diameter << char_t(' ') << rcrd.extent << char_t(' ') << rcrd.solidity
                    << char_t(' ') << rcrd.roundness << char_t(' ') << rcrd.compactness << char_t(' ') << rcrd.shape_factor_1 << char_t(' ')
                    << rcrd.shape_factor_2 << char_t(' ') << rcrd.shape_factor_3 << char_t(' ') << rcrd.shape_factor_4 << char_t('\n');
            return ostream;
        }
};

template<std::floating_point T> static constexpr record<T> parse_line(const std::string_view& line) noexcept {
    // a typical row will be in the format of,
    // 28395,610.291,208.178116708527,173.888747041636,1.19719142411602,0.549812187138347,28715,190.141097274511,0.763922518159806,0.988855998607,0.958027126250128,0.913357754795763,0.00733150613518321,0.00314728916733569,0.834222388245556,0.998723889013168,SEKER
    record<T> temporary {};

    const char* const cstart { line.data() };
    const char*       begin { line.data() };

    size_t caret = line.find(',', 0); // the first comma
    std::from_chars(begin, begin + caret, temporary.area);
    begin += caret;

    caret  = line.find(',', caret + 1); // the next coma
    std::from_chars(begin, /* char next to the comma */ begin + caret, temporary.perimeter);
    begin += caret;

    std::from_chars(begin += 2, begin += line.find(',', std::distance(cstart, begin)) - 1, temporary.major_axis_length);
    std::from_chars(begin += 2, begin += line.find(',', std::distance(cstart, begin)) - 1, temporary.minor_axis_length);
    std::from_chars(begin += 2, begin += line.find(',', std::distance(cstart, begin)) - 1, temporary.aspect_ratio);
    std::from_chars(begin += 2, begin += line.find(',', std::distance(cstart, begin)) - 1, temporary.eccentricity);
    std::from_chars(begin += 2, begin += line.find(',', std::distance(cstart, begin)) - 1, temporary.convex_area);
    std::from_chars(begin += 2, begin += line.find(',', std::distance(cstart, begin)) - 1, temporary.equiv_diameter);
    std::from_chars(begin += 2, begin += line.find(',', std::distance(cstart, begin)) - 1, temporary.extent);
    std::from_chars(begin += 2, begin += line.find(',', std::distance(cstart, begin)) - 1, temporary.roundness);
    std::from_chars(begin += 2, begin += line.find(',', std::distance(cstart, begin)) - 1, temporary.compactness);
    std::from_chars(begin += 2, begin += line.find(',', std::distance(cstart, begin)) - 1, temporary.extent);
    std::from_chars(begin += 2, begin += line.find(',', std::distance(cstart, begin)) - 1, temporary.shape_factor_1);
    std::from_chars(begin += 2, begin += line.find(',', std::distance(cstart, begin)) - 1, temporary.shape_factor_2);
    std::from_chars(begin += 2, begin += line.find(',', std::distance(cstart, begin)) - 1, temporary.shape_factor_3);
    std::from_chars(begin += 2, begin += line.find(',', std::distance(cstart, begin)) - 1, temporary.shape_factor_4);

    return temporary;
}

auto wmain() -> int {
    const auto test = ::parse_line<double>(
        "28395,610.291,208.178116708527,173.888747041636,1.19719142411602,0.549812187138347,28715,190.141097274511,0.763922518159806,0.988855998607,0.958027126250128,0.913357754795763,0.00733150613518321,0.00314728916733569,0.834222388245556,0.998723889013168,SEKER"
    );

    std::wcout << test;

    return EXIT_SUCCESS;
}
