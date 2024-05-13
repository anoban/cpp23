#include <cstddef>
#include <cstdlib>
#include <string>
#include <type_traits>

template<
    typename char_t,
    std::size_t default_capacity = 100,
    typename                     = std::enable_if<std::is_same<char, char_t>::value || std::is_same<wchar_t, char_t>::value, char_t>::type>
class string {
        using value_type         = char_t;
        using pointer_type       = char_t*;
        using const_literal_type = const char_t* const;
        using iterator           = char_t*;
        using const_iterator     = const char_t*;

    public:
        explicit string() noexcept : buffer { ::malloc(default_capacity * sizeof(char_t)) } { }

        explicit string(const_literal_type str) noexcept { }

        ~string() noexcept {
            ::free(buffer);
            buffer = nullptr;
            length = capacity = 0;
        }

    private:
        pointer_type buffer {};
        std::size_t  length {};
        std::size_t  capacity {};
};

template<typename T> static constexpr typename std::basic_string<T> concatenate(const T&) noexcept { }

int wmain() { }
