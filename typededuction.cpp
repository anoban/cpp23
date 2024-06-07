#include <cstdlib>
#include <iostream>

template<typename T> [[nodiscard]] T func(T _inp) noexcept {
#if defined(__llvm__) && defined(__clang__)
    std::wcout << __PRETTY_FUNCTION__ << L'\n';
#elif defined(_MSC_FULL_VER)
    std::wcout << __FUNCSIG__ << L'\n';
#endif
    return _inp;
}

// templated function return types decay to its underlying types!

int main() {
    const int        cint    = 76475; // a constant integer
    const int&       refcint = cint;  // reference to a constant integer
    const int*       ptr     = &cint; // pointer to a constant integer
    const int* const cptr    = &cint; // constant pointer to a constant integer

    func(cint);
    func(refcint);
    func(ptr);
    func(cptr);

    return EXIT_SUCCESS;
}
