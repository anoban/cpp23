#include <cstdlib>
#include <iostream>

namespace endian {

    // clang-format off
#include <intrin.h> // __m64
// MSVC does not support _mm_shuffle_pi8 in x64 architectures
    // clang-format on

    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    static constexpr unsigned short __stdcall ushort_from_be_bytes(_In_reads_bytes_(2) const unsigned char* const bytestream) noexcept {
        return static_cast<unsigned>(bytestream[1]) | static_cast<unsigned>(bytestream[0]) >> 8;
    }

    static unsigned long __stdcall ulong_from_be_bytes(_In_reads_bytes_(4) const unsigned char* const bytestream) noexcept {
#if defined(__llvm__) && defined(__clang__)
        static constexpr __m64 mask_pi8 { 0b00000011'00000010'00000001'00000000'00000100'00000101'00000110'00000111 };
        return endian::_mm_shuffle_pi8(
            *reinterpret_cast<const __m64*>(bytestream), mask_pi8
        )[0]; // LLVM defines __m64 as a vector of 1 long long
#elif defined(_MSC_VER) && defined(_MSC_FULL_VER)
        const __m128i operand_epi8 {
            .m128i_u8 {
                       bytestream[0], bytestream[1], bytestream[2], bytestream[3], /* 12 filler zeroes */ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
        };
        static constexpr __m128i mask_epi8 {
            .m128i_u8 { 3, 2, 1, 0, /* we don't care about the rest */ 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 }
        };
        // unlike LLVM, MSVC offers the intrinsic _mm_shuffle_pi8 only in x86 mode
        return endian::_mm_shuffle_epi8(operand_epi8, mask_epi8).m128i_u32[0]; // MSVC defines __m128i as a union
#endif
    }

    static unsigned long long __stdcall ullong_from_be_bytes(_In_reads_bytes_(8) const unsigned char* const bytestream) noexcept {
        static constexpr __m64 mask { 0b00000000'00000001'00000010'00000011'00000100'00000101'00000110'00000111 };
#if defined(__llvm__) && defined(__clang__)
        return endian::_mm_shuffle_pi8(*reinterpret_cast<const __m64*>(bytestream), mask)[0];
#elif defined(_MSC_VER) && defined(_MSC_FULL_VER)
        const __m128i operand_epi8 {
            .m128i_u8 { bytestream[0],
                       bytestream[1],
                       bytestream[2],
                       bytestream[3],
                       bytestream[4],
                       bytestream[5],
                       bytestream[6],
                       bytestream[7],
                       0, /* 8 filler zeroes */
                        0, 0,
                       0, 0,
                       0, 0,
                       0 }
        };
        static constexpr __m128i mask_epi8 {
            .m128i_u8 { 7, 6, 5, 4, 3, 2, 1, 0, /* we don't care about the rest */ 8, 9, 10, 11, 12, 13, 14, 15 }
        };

        return endian::_mm_shuffle_epi8(operand_epi8, mask_epi8).m128i_u64[0];
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

    std::wcout << L"----------------------------------\n";

    std::wcout << endian::ushort_from_be_bytes(bytes) << L'\n';
    std::wcout << endian::ulong_from_be_bytes(bytes) << L'\n';
    std::wcout << endian::ullong_from_be_bytes(bytes) << L'\n';

    return EXIT_SUCCESS;
}
