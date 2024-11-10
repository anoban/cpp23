#include <cstdlib>
#include <iostream>

namespace endian {

    // clang-format off
#define _M_IX86 // needed by <intrin.h>
#define _M_X64  // needed by <intrin.h>
#include <mmintrin.h> // __m64
#include <intrin.h>
    // clang-format on

    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    static constexpr unsigned short __stdcall ushort_from_be_bytes(_In_reads_bytes_(2) const unsigned char* const bytestream) noexcept {
        return static_cast<unsigned short>(bytestream[0]) >> 8 | (bytestream[1]) << 8;
    }

    static unsigned long __stdcall ulong_from_be_bytes(_In_reads_bytes_(4) const unsigned char* const bytestream) noexcept {
        static constexpr __m64 mask { 0b00000100'00000101'00000110'00000111'0000000'00000001'00000010'00000011 };
#if defined(__llvm__) && defined(__clang__)
        return endian::_mm_shuffle_pi8(
            *reinterpret_cast<const __m64*>(bytestream), mask
        )[0]; // LLVM defines __m64 as a vector of 1 long long
#elif defined(_MSC_VER) && defined(_MSC_FULL_VER)
        return endian::_mm_shuffle_pi8(*reinterpret_cast<const __m64*>(bytestream), mask).m64_u32[0]; // MSVC defines __m64 as a union
#endif
    }

    static unsigned long long __stdcall ullong_from_be_bytes(_In_reads_bytes_(8) const unsigned char* const bytestream) noexcept {
        static constexpr __m64 mask { 0b0000000'00000001'00000010'00000011'00000100'00000101'00000110'00000111 };
#if defined(__llvm__) && defined(__clang__)
        return endian::_mm_shuffle_pi8(*reinterpret_cast<const __m64*>(bytestream), mask)[0];
#elif defined(_MSC_VER) && defined(_MSC_FULL_VER)
        return endian::_mm_shuffle_pi8(*reinterpret_cast<const __m64*>(bytestream), mask).m64_u64;
#endif
    }
    // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)

} // namespace endian

auto wmain() -> int {
    static constexpr unsigned char bytes[] { 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08 };

    std::wcout << std::hex << std::uppercase;
    std::wcout << *reinterpret_cast<const unsigned short*>(bytes) << L'\n';
    std::wcout << *reinterpret_cast<const unsigned*>(bytes) << L'\n';
    std::wcout << *reinterpret_cast<const unsigned long long*>(bytes) << L'\n';

    return EXIT_SUCCESS;
}
