#include <concepts>
#include <iostream>
#include <vector>

constexpr std::size_t NCHARS_SPECIESNAME { 20 }; // Iris-versicolor is the longest species name and it has 15 characters so 20 is okay :)
constexpr std::size_t NROWS_CSV { 200 };         // iris.data has only 150 rows

// columns : Sepal Length, Sepal Width, Petal Length, Petal Width & Species

template<typename T> requires std::floating_point<T> struct record {
        T    sepal_length {};
        T    sepal_width {};
        T    petal_length {};
        T    petal_width {};
        char species[NCHARS_SPECIESNAME] {};

        constexpr record() = default;
};

template<typename T> requires std::floating_point<T> class Iris {
    private:
        std::array<record<T>, NROWS_CSV> rows {};
};
