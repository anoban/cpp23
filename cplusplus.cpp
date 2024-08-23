#include <cstdio>
#include <cstdlib>
#pragma warning(disable : 4514)

// https://learn.microsoft.com/en-us/cpp/build/reference/zc-cplusplus?view=msvc-170

int wmain() {
    ::wprintf_s(
        L"__cplusplus is %ld\n", __cplusplus
    ); // MSVC requires /Zc:__cplusplus compiler flag to define the correct __cplusplus macro
    // by default MSVC always defines __cplusplus to 199711 regardless of the /std:c++xx flag
    return EXIT_SUCCESS;
}
