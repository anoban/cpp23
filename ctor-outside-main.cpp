// g++ ctor-outside-main.cpp -Wall -Wextra -Wpedantic -O3 -std=c++20 -municode

#include <cstdlib>
#include <iostream>
#include <ostream>
#include <type_traits>

template<typename T, bool = std::is_arithmetic_v<T>> struct object;

struct some { };

template<typename T> concept is_character = std::is_same_v<T, char> || std::is_same_v<T, wchar_t>;

template<typename T> struct object<T, true> final { // a partial overload
    private:
        T value;

    public:
        object() noexcept : value(11) { } // non-constexpr ctor

        T operator()() const noexcept { return value; }

        template<typename char_t> requires ::is_character<char_t>
        friend std::basic_ostream<char_t>& operator<<(std::basic_ostream<char_t>& ostream, const object& obj) {
            ostream << obj.value << char_t('\n');
            return ostream;
        }
};

object<float> obj; // initialization of this object requires a call to its ctor
// how does that happen outside of main?
// C++ object model gurantees that variable obj will be initialized before its first use in main
// this doesn't mean obj will be initialized at the startup of main()!

// object<some> undefined_template;

// a compile time evaluated function
template<typename T> static consteval T func(const object<T>& obj) noexcept requires std::is_arithmetic_v<T> { return obj(); }

// constexpr auto value { ::func(obj) }; // obj is not a compile time constant
// object.operator() is not constexpr

template<typename T> requires std::is_arithmetic_v<T> struct objectv2 final {
    private:
        T value;

    public:
        using value_type = T;

        constexpr objectv2() noexcept : value(22) { }

        constexpr T operator()() const noexcept { return value; }

        template<typename char_t> requires ::is_character<char_t>
        friend std::basic_ostream<char_t>& operator<<(std::basic_ostream<char_t>& ostream, const objectv2& obj) {
            ostream << obj.value << char_t('\n');
            return ostream;
        }
};

template<typename T> requires std::is_arithmetic_v<T> struct objectv3 final {
    private:
        T value;

    public:
        using value_type = T;

        constexpr objectv3() noexcept : value(33) { }

        template<typename char_t> requires ::is_character<char_t>
        friend std::basic_ostream<char_t>& operator<<(std::basic_ostream<char_t>& ostream, const objectv3& obj) {
            ostream << obj.value << char_t('\n');
            return ostream;
        }
};

template<typename T> static consteval T func(const objectv2<T>& obj) noexcept requires std::is_arithmetic_v<T> { return obj(); }

constexpr objectv2<short> objs;

constexpr auto value2 { ::func(objs) }; // works because objs is constexpr and operator() is also constexpr

namespace f {
    template<typename T> requires std::is_class_v<T> [[nodiscard]] static consteval auto func(const T& obj
                                  ) noexcept -> decltype(obj.operator()()) requires requires { obj.operator()(); } {
        return obj();
    }

    template<typename T> requires std::is_class_v<T> [[nodiscard]] static consteval auto fn(const T& obj) noexcept -> typename T::value_type
                                  requires requires { obj.operator()(); } {
        return obj();
    }
} // namespace f

constexpr auto value3 { f::func(objs) };

constexpr objectv3<double> objv3;

constexpr auto value4 { f::fn(objv3) };

int wmain() {
    std::wcout << obj;

    std::wcout << objs;
    return EXIT_SUCCESS;
}
