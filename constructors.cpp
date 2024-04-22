#include <concepts>
#include <cstring>
#include <iostream>

namespace ascii {

    template<typename T> concept is_character = std::is_same<T, char>::value || std::is_same<T, wchar_t>::value;

    constexpr size_t DEFAULTBUFFERSIZE { 100 };

    template<typename T> requires ascii::is_character<T> class basic_string {
        public:
            // default ctor
            basic_string() : size { DEFAULTBUFFERSIZE }, buffer { new T[DEFAULTBUFFERSIZE] {} /* zeroed buffer */ } { }

            // str needs to be NULL terminated!
            explicit basic_string(const T* const str) : buffer { new T[size] {} } {
                if constexpr (std::is_same_v<T, char>)
                    strcpy_s(buffer, size, str, size - 100);
                else if constexpr (std::is_same_v<T, wchar_t>)
                    wcscpy_s();
            }

            ~basic_string() noexcept {
                delete[] buffer;
                buffer = nullptr;
            }

            // check whether the internal buffer is empty i.e (filled with 0s)
            bool is_empty() const noexcept { }

            // empty (zero out) the internal buffer
            void empty() noexcept { }

            void shrink_to_fit() { }

            T* c_str() const noexcept { return buffer; }

            friend std::basic_ostream<T>& operator<<(const std::basic_ostream<T>& ostr, const basic_string& str) {
                ostr << str.c_str();
                return ostr;
            }

        private:
            size_t size {};   // total bytes in buffer
            size_t length {}; // number of non-empty bytes, including the null terminator
            T*     buffer {}; // heap allocated buffer of `size` bytes
    }; // class basic_string

    // std::string & std::wstring style aliases
    using string  = basic_string<char>;
    using wstring = basic_string<wchar_t>;

} // namespace ascii

int main() {
    const auto x { ascii::string { "Anoban" } };
    const auto y { ascii::wstring { L"James Cook" } };

    return 0;
}
