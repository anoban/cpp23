#include <cstdio>
#include <cstdlib>
#pragma warning(disable : 4514)

int wmain() {
    ::wprintf_s(L"__cplusplus is %ld\n", __cplusplus);
    return EXIT_SUCCESS;
}
