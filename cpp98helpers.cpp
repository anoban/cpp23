// g++ cpp98helpers.cpp -Wall -Wextra -O3 -std=c++98 -Wpedantic

#include <algorithm>
#include <array>
#include <ctime>
#include <functional>
#include <iostream>

struct randf {
        void operator()(float* f) const throw() { *f = static_cast<float>(rand()) / RAND_MAX; }
};

int main() {
    srand(time(NULL));
    // std::array<float, 100> arr; std::array is a C++11 feature
    float       arr[100];

    // std::begin and std::end are C++11 features
    const randf rng; // const randf rng(); gets interpreted as a function declaration
    for (float *it = arr, *end = (arr + 100); it != end; ++it) rng.operator()(it);

    // even with g++, std::bind2nd requires -std=c++11 and gives a deprecated warning
    size_t count = std::count_if(arr, arr + 100, std::bind2nd(std::less<float>(), 0.50000));
    std::wcout << L"There were " << count << L" elements lesser than 0.5\n";

    count = 0;
    for (unsigned i = 0; i < 100; ++i) count += (arr[i] < 0.5000);
    std::wcout << L"There were " << count << L" elements lesser than 0.5\n";

    return EXIT_SUCCESS;
}
