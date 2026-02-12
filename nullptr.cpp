#include <cstdio>
#include <cstdlib>

static void foo(const int x) noexcept { ::puts("foo(const int)"); }

static void foo(const void* x) noexcept { ::puts("foo(const void*)"); }

int main() {
    foo(NULL); // some implementations define NULL as ((void*) 0)
    // so in certain systems foo(NULL) could invoke the second overload

    foo(nullptr); // always invokes the second overload

    foo(static_cast<void*>(0));

    auto type { nullptr }; // type of nullptr is std::nullptr

    if (type) ::puts(":)");

    unsigned long long x {
        nullptr
    }; // Error: cannot initialize a variable of type 'unsigned long long' with an rvalue of type 'std::nullptr_t'

    return EXIT_SUCCESS;
}
