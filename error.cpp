#pragma once
// clang-format off
#define _AMD64_
#define WIN32_LEAN_AND_MEAN
#define WIN32_EXTRA_MEAN
#define NOMINMAX

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <vector>
#include <optional>
#include <Windows.h>
// clang-format on

namespace bmp {
    [[nodiscard("expensive")]] static inline std::optional<std::vector<uint8_t>> Open(_In_ const wchar_t* const filename) noexcept {
        DWORD                nbytes {};
        LARGE_INTEGER        liFsize = { .QuadPart = 0LLU };
        const HANDLE64       hFile   = ::CreateFileW(filename, GENERIC_READ, 0, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_READONLY, NULL);
        std::vector<uint8_t> buffer;

        if (hFile == INVALID_HANDLE_VALUE) {
            ::fprintf(stderr, L"Error %lu in CreateFileW\n", ::GetLastError()); // NOLINT(cppcoreguidelines-pro-type-vararg)
            goto INVALID_HANDLE_ERR;
        }

        if (!::GetFileSizeEx(hFile, &liFsize)) {
            ::fprintf(stderr, L"Error %lu in GetFileSizeEx\n", ::GetLastError()); // NOLINT(cppcoreguidelines-pro-type-vararg)
            goto GET_FILESIZE_ERR;
        }

        buffer.resize(liFsize.QuadPart);

        if (!::ReadFile(hFile, buffer.data(), liFsize.QuadPart, &nbytes, NULL)) {
            ::fprintf(stderr, L"Error %lu in ReadFile\n", ::GetLastError()); // NOLINT(cppcoreguidelines-pro-type-vararg)
            goto GET_FILESIZE_ERR;
        }

        ::CloseHandle(hFile);
        return buffer;

GET_FILESIZE_ERR:
        CloseHandle(hFile);
INVALID_HANDLE_ERR:
        return std::nullopt;
    }

    static inline BITMAPFILEHEADER parsefileheader(_In_ const std::vector<uint8_t>& imstream) noexcept {
        assert(imstream.size() >= sizeof(BITMAPFILEHEADER));

        BITMAPFILEHEADER header { .bfType = 0, .bfSize = 0, .bfReserved1 = 0, .bfReserved2 = 0, .bfOffBits = 0 };

        header.bfType = (((uint16_t) (*(imstream.data() + 1))) << 8) | ((uint16_t) (*imstream.data()));
        if (header.bfType != (((uint16_t) 'M' << 8) | (uint16_t) 'B')) {
            fputws(L"Error in parsefileheader, file appears not to be a Windows BMP file\n", stderr);
            return header;
        }

        header.bfSize    = *reinterpret_cast<const uint32_t*>(imstream.data() + 2);
        header.bfOffBits = *reinterpret_cast<const uint32_t*>(imstream.data() + 10);

        return header;
    }

    static inline BITMAPINFOHEADER parseinfoheader(_In_ const std::vector<uint8_t>& imstream) noexcept {
        assert(imstream.size() >= (sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER)));

        BITMAPINFOHEADER header {};

        if (*reinterpret_cast<const uint32_t*>(imstream.data() + 14U) > 40U) { // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
            fputws(L"Error in parseinfoheader, BMP image seems to contain an unparsable file info header", stderr);
            return header;
        }

        // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        header.biSize          = *(reinterpret_cast<const uint32_t*>(imstream.data() + 14U));
        header.biWidth         = *(reinterpret_cast<const uint32_t*>(imstream.data() + 18U));
        header.biHeight        = *(reinterpret_cast<const int32_t*>(imstream.data() + 22U));
        header.biPlanes        = *(reinterpret_cast<const uint16_t*>(imstream.data() + 26U));
        header.biBitCount      = *(reinterpret_cast<const uint16_t*>(imstream.data() + 28U));
        header.biCompression   = *(reinterpret_cast<const uint32_t*>(imstream.data() + 30U));
        header.biSizeImage     = *(reinterpret_cast<const uint32_t*>(imstream.data() + 34U));
        header.biXPelsPerMeter = *(reinterpret_cast<const uint32_t*>(imstream.data() + 38U));
        header.biYPelsPerMeter = *(reinterpret_cast<const uint32_t*>(imstream.data() + 42U));
        header.biClrUsed       = *(reinterpret_cast<const uint32_t*>(imstream.data() + 46U));
        header.biClrImportant  = *(reinterpret_cast<const uint32_t*>(imstream.data() + 50U));
        // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)

        return header;
    }

} // namespace bmp
