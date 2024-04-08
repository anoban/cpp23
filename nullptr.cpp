#include <cstdio>
#include <cstdlib>

static void foo(const int x) noexcept { ::_putws(L"foo(const int)"); }

static void foo(const void* x) noexcept { ::_putws(L"foo(const void*)"); }

int         main() {
    foo(NULL); // some implementations define NULL as ((void*) 0)
    // so in certain systems foo(NULL) could invoke the second overload

    foo(nullptr); // always invokes the second overload

    foo(static_cast<void*>(0));

    return EXIT_SUCCESS;
}
