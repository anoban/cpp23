#include <charconv>
#include <iomanip>
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

        friend std::ostream& operator<<(_Inout_ std::ostream& ostream, _In_ const record& rcrd) noexcept {
            ostream << rcrd.area << ' ' << rcrd.perimeter << ' ' << rcrd.major_axis_length << ' ' << rcrd.minor_axis_length << ' '
                    << rcrd.aspect_ratio << ' ' << rcrd.eccentricity << ' ' << rcrd.convex_area << ' ' << rcrd.equiv_diameter << ' '
                    << rcrd.extent << ' ' << rcrd.solidity << ' ' << rcrd.roundness << ' ' << rcrd.compactness << ' ' << rcrd.shape_factor_1
                    << ' ' << rcrd.shape_factor_2 << ' ' << rcrd.shape_factor_3 << ' ' << rcrd.shape_factor_4 << '\n';
            return ostream;
        }
};

template<std::floating_point T> static constexpr record<T> parse_line(_In_ const std::string_view& line) noexcept {
    // a typical row will be in the format of,
    // 28395,610.291,208.178116708527,173.888747041636,1.19719142411602,0.549812187138347,28715,190.141097274511,0.763922518159806,0.988855998607,0.958027126250128,0.913357754795763,0.00733150613518321,0.00314728916733569,0.834222388245556,0.998723889013168,SEKER
    record<T>         temporary {};
    const char* const cstart { line.data() };
    const char*       begin { line.data() };

    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)

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

    // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)

    return temporary;
}

auto main() -> int {
    std::cout << std::setprecision(15);

    const auto test = ::parse_line<long double>(
        "28395,610.291,208.178116708527,173.888747041636,1.19719142411602,0.549812187138347,28715,190.141097274511,0.763922518159806,0.988855998607,0.958027126250128,0.913357754795763,0.00733150613518321,0.00314728916733569,0.834222388245556,0.998723889013168,SEKER"
    );

    std::cout << test;

    return EXIT_SUCCESS;
}
