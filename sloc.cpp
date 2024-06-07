#include <iostream>
#include <source_location>
#include <type_traits>

#include <sal.h>

#include <source_location.hpp>


template<typename char_t>
static typename std::enable_if_t<std::is_arithmetic_v<char_t>, void> logger(
    _In_ const std::source_location& _sloc = std::source_location::current()
) {
    std::cout << "File name :: " << _sloc.file_name() << "\nFunction name :: " << _sloc.function_name()
              << "\nLine number :: " << _sloc.line() << "\nColumn number :: " << _sloc.column() << std::endl;
}

int wmain() {
    logger<float>();

    std::cout << std::endl;
    std::cout << std::endl;

    constexpr auto current = experimental::source_location<char>::current();
    std::cout << "File name :: " << current.file_name() << "\nFunction name :: " << current.function_name()
              << "\nFull function signature :: " << current.function_signature() << "\nLine number :: " << current.line()
              << "\nColumn number :: " << current.column() << std::endl;

    constexpr auto wcurrent = experimental::source_location<wchar_t>::current();

    return EXIT_SUCCESS;
}
