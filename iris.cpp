// g++ iris.cpp -Wall -Wextra -O3 -std=c++20 -Wpedantic -municode
// -municode is critical for the use of wmain

#include <concepts>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <sal.h>

namespace iris {
    // Iris-versicolor is the longest species name and it has 15 characters so 20 is okay :)
    constexpr std::size_t NCHARS_SPECIESNAME { 20 }; // ::wcsnlen(L"Iris-versicolor", 16) + 5
    constexpr std::size_t NROWS_CSV { 150 };         // iris.data has only 150 rows

    // columns : Sepal Length, Sepal Width, Petal Length, Petal Width & Species
    template<typename T> struct record {
            typename std::enable_if<std::is_floating_point<T>::value, T>::type sepal_length {};
            typename std::enable_if<std::is_floating_point<T>::value, T>::type sepal_width {};
            typename std::enable_if<std::is_floating_point<T>::value, T>::type petal_length {};
            typename std::enable_if<std::is_floating_point<T>::value, T>::type petal_width {};
            std::string                                                        species;

            record() = delete; // no default ctors

            record(const T sl, const T sw, const T pl, const T pw, std::string& sp) noexcept :
                sepal_length { sl }, sepal_width { sw }, petal_length { pl }, petal_width { pw }, species { std::move(sp) } { }

            ~record() = default;
    };

    std::string read_file(_In_ const wchar_t* const filename) {
        std::ifstream     ifile { filename, std::ios::in };
        std::stringstream sstream {};
        sstream << ifile.rdbuf();
        ifile.close();
        return sstream.str();
    }

} // namespace iris

auto wmain(_In_opt_ int argc, _In_opt_ wchar_t* argv[]) -> int {
    if (argc == 1 || argc > 2) ::exit(11);
    const auto contents { iris::read_file(argv[1]) };

    std::vector<iris::record<float>> records;
    size_t                           newline_caret {};
    size_t                           nlines {};
    while ((newline_caret = contents.find('\n', newline_caret + 1)) != std::string::npos) {
        // parse the line
        std::cout << newline_caret << '\n';
        nlines++;
    }
    return EXIT_SUCCESS;
}
