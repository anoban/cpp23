#include <type_traits>

template<unsigned offset, class T, class... TList> struct get_nth final {
        using type = get_nth<offset - 1, TList...>::type;
};

template<class T, class... TList> struct get_nth<0, T, TList...> final {
        using type = T;
};

template<unsigned offset, class... TList> using get_nth_t = typename /* cannot use class here! */ ::get_nth<offset, TList...>::type;

static_assert(!std::is_same_v<::get_nth<2, wchar_t, long long, volatile short&&, unsigned, const int, long&>::type, unsigned>);
static_assert(std::is_same_v<::get_nth<3, wchar_t, long long, volatile short&&, unsigned, const int, long&>::type, unsigned>);
static_assert(std::is_same_v<::get_nth<1, float, double, volatile long double&&>::type, double>);
static_assert(std::is_same_v<::get_nth<0, char, double, volatile long double&&, float&>::type, char>);

static_assert(!std::is_same_v<::get_nth_t<2, wchar_t, long long, volatile short&&, unsigned, const int, long&>, unsigned>);
static_assert(std::is_same_v<::get_nth_t<3, wchar_t, long long, volatile short&&, unsigned, const int, long&>, unsigned>);
static_assert(std::is_same_v<::get_nth_t<1, float, double, volatile long double>, double>);
static_assert(std::is_same_v<::get_nth_t<0, char, double, volatile long double&&, float&>, char>);
