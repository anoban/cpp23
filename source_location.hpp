#include <cstdint>
#include <type_traits>

template<typename char_t> concept is_iostream_compatible = std::is_same_v<char, char_t> || std::is_same_v<wchar_t, char_t>;

#define __builtin_FUNCSIGW()  (L##__FUNCSIG__)
#define __builtin_FUNCTIONW() (__FUNCTIONW__)

namespace experimental {

    template<typename char_t> requires ::is_iostream_compatible<char_t> struct source_location;

    template<> struct source_location<char> final {
            [[nodiscard]] static consteval source_location current(
                const unsigned _Line_           = __builtin_LINE(),
                const unsigned _Column_         = __builtin_COLUMN(),
                const char* const _File_        = __builtin_FILE(),
                const char* const _FunctionSig_ = __builtin_FUNCSIG(),
                const char* const _Function_    = __builtin_FUNCTION()
            ) noexcept {
                source_location _Result { _Line_, _Column_, _File_, _FunctionSig_, _Function_ };
                return _Result;
            }

            [[nodiscard]] constexpr source_location() noexcept = default;

            [[nodiscard]] constexpr source_location(
                const unsigned    _Line_,
                const unsigned    _Column_,
                const char* const _File_,
                const char* const _FunctionSig_,
                const char* const _Function_
            ) noexcept :
                _Line { _Line_ }, _Column { _Column_ }, _File { _File_ }, _FunctionSig { _FunctionSig_ }, _Function { _Function_ } { }

            [[nodiscard]] constexpr unsigned    line() const noexcept { return _Line; }

            [[nodiscard]] constexpr unsigned    column() const noexcept { return _Column; }

            [[nodiscard]] constexpr const char* file_name() const noexcept { return _File; }

            [[nodiscard]] constexpr const char* function_name() const noexcept { return _Function; }

            [[nodiscard]] constexpr const char* function_signature() const noexcept { return _FunctionSig; }

        private:
            const unsigned    _Line {};
            const unsigned    _Column {};
            const char* const _File {};
            const char* const _FunctionSig {};
            const char* const _Function {};
    };

    template<> struct source_location<wchar_t> final {
            [[nodiscard]] static consteval source_location current(
                const unsigned _Line_              = __builtin_LINE(),
                const unsigned _Column_            = __builtin_COLUMN(),
                const wchar_t* const _File_        = __FILEW__,
                const wchar_t* const _FunctionSig_ = __builtin_FUNCSIGW(),
                const wchar_t* const _Function_    = __builtin_FUNCTIONW()
            ) noexcept {
                source_location _Result { _Line_, _Column_, _File_, _FunctionSig_, _Function_ };
                return _Result;
            }

            [[nodiscard]] constexpr source_location() noexcept = default;

            [[nodiscard]] constexpr source_location(
                const unsigned       _Line_,
                const unsigned       _Column_,
                const wchar_t* const _File_,
                const wchar_t* const _FunctionSig_,
                const wchar_t* const _Function_
            ) noexcept :
                _Line { _Line_ }, _Column { _Column_ }, _File { _File_ }, _FunctionSig { _FunctionSig_ }, _Function { _Function_ } { }

            [[nodiscard]] constexpr unsigned       line() const noexcept { return _Line; }

            [[nodiscard]] constexpr unsigned       column() const noexcept { return _Column; }

            [[nodiscard]] constexpr const wchar_t* file_name() const noexcept { return _File; }

            [[nodiscard]] constexpr const wchar_t* function_name() const noexcept { return _Function; }

            [[nodiscard]] constexpr const wchar_t* function_signature() const noexcept { return _FunctionSig; }

        private:
            const unsigned       _Line {};
            const unsigned       _Column {};
            const wchar_t* const _File {};
            const wchar_t* const _FunctionSig {};
            const wchar_t* const _Function {};
    };

} // namespace experimental
