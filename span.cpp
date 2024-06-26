#include <iostream>
#include <span>

template<typename char_t> std::basic_ostream<char_t>& operator<<(std::basic_ostream<char_t>& ostream, const std::byte& byte) {
    ostream << std::hex << std::uppercase; // inefficient to do this for every byte!
    ostream << static_cast<uint8_t>(byte);
    return ostream;
}

auto wmain() -> int {
    //
    constexpr unsigned numbers[4] { 0xFF00FFAA, 0x00AABBCC, 0xFFDDEEFF, 0x44332211 }; // remember the endianness!
    std::span          spnum { numbers };
    for (const auto& b : std::as_bytes(spnum)) std::wcout << b << L'\n';

    // auto bview { std::as_bytes(numbers) };
    // std::as_bytes() only accepts a std::span object as input!

    return EXIT_SUCCESS;
}
