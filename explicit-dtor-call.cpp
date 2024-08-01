#include <string>
// clang-format off
            using std::wstring;
// clang-format on

auto wmain() -> int {
    [[maybe_unused]] const auto billie { wstring { L"Was it obvious to everybody else?" } };
    const auto* const           _ptr { billie.data() };

    ::wprintf_s(L"size :: %4zu, capacity :: %4zu, address :: %08X\n", billie.size(), billie.capacity(), _ptr);
    billie.~basic_string(); // explicitly calling the destructor does not free the string's internal buffer
    ::wprintf_s(L"size :: %4zu, capacity :: %4zu, address :: %08X\n", billie.size(), billie.capacity(), _ptr);

    return EXIT_SUCCESS;
}
