#include <cstdlib>
#include <string>

static inline const wchar_t* func() throw() {
    std::wstring greeting {
        L"Let's hope this gets allocated on the heap, making this gratuitously long just to make sure it doesn't get placed on the stack! :)"
    };
    return greeting.data();
}

// let's try and take advantage of move semantics
static inline const wchar_t* move() throw() {
    std::wstring greeting {
        L"Let's hope this gets allocated on the heap, making this gratuitously long just to make sure it doesn't get placed on the stack! :)"
    };
    return std::move(greeting).data(); // we are using a xvalue here!
}

auto wmain() -> int {
    const auto str { ::func() };
    ::_putws(str);

    const auto mstr { ::move() };
    ::_putws(mstr);
    return EXIT_SUCCESS;
}
