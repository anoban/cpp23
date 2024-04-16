#include <concepts>
#include <fstream>
#include <iostream>
#include <vector>

namespace iris {

    constexpr std::size_t NCHARS_SPECIESNAME {
        20
    }; // Iris-versicolor is the longest species name and it has 15 characters so 20 is okay :)
    constexpr std::size_t NROWS_CSV { 150 }; // iris.data has only 150 rows

    // columns : Sepal Length, Sepal Width, Petal Length, Petal Width & Species

    template<typename T> requires std::floating_point<T> struct record {
            T    sepal_length {};
            T    sepal_width {};
            T    petal_length {};
            T    petal_width {};
            char species[NCHARS_SPECIESNAME] {};
    };

    template<typename T> static std::vector<record<T>> readIrisData(_In_ const wchar_t* const filename) noexcept {
        std::ifstream ifile { L"./iris.data", std::ios::in };
        std::istreambuf_iterator<char>(ifile.tellg());
    }

} // namespace iris
