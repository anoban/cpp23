#include <memory>

static void release(void* _ptr) noexcept(noexcept(::free(_ptr))) {
    ::free(_ptr);
    _ptr = nullptr;
}

auto wmain() -> int {
    auto* memory = ::malloc(18783);
    ::release(memory);
    *reinterpret_cast<int*>(memory) = 46818968; // access violation error

    return EXIT_SUCCESS;
}
