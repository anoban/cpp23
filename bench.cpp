#include <algorithm>
#include <execution>
#include <iostream>
#include <random>
#include <vector>

#include <Windows.h>

#define LENGTH 1000'000LLU

int wmain() {
    LARGE_INTEGER start = { .QuadPart = 0 }, stop = { .QuadPart = 0 };
    QueryPerformanceCounter(&start);

    auto rdevice { std::random_device {} };
    auto rengine { std::mt19937_64 { rdevice() } };

    std::vector<int32_t> randoms(LENGTH);
    std::ranges::generate(randoms, rengine);

    const auto count =
        std::count_if(std::execution::seq, randoms.cbegin(), randoms.cend(), [](const int& x) -> bool { return x < (INT32_MAX / 2); });
    std::wcout << count << L'\n';

    QueryPerformanceCounter(&stop);
    std::wcout << L"Execution took " << stop.QuadPart - start.QuadPart << L" ticks!\n";
    return EXIT_SUCCESS;
}
