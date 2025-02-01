#pragma once

#include <chrono>
#include <cstdio>
#include <cstdlib>

class csv_logger final {
    private:
        FILE               _handle;
        bool               _is_open;
        bool               _has_header;
        const char*        _header;
        unsigned long long _ncolumns;
        unsigned long long _nrows;

    public:
        template<unsigned long long _len> bool __cdecl open(const char (&_filename)[_len]) noexcept { }

        template<unsigned long long _len> bool __cdecl open(const wchar_t (&_filename)[_len]) noexcept { }

        void push_back_comma() noexcept { }

        void push_back_space() noexcept { }

        void push_back_newline() noexcept { }
};
