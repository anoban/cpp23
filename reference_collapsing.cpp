#include <type_traits>

// programmers are not allowed to declare reference to references directly in C++!!

struct pod {
        long   _long;
        double _double;
};

static inline constexpr pod get() noexcept { return { 12, 100.0 }; }

template<typename T> using add_lvalue_reference_t = T&;
template<typename T> using add_rvalue_reference_t = T&&;

auto wmain() -> int {
    //

    int  hundred { 100 };
    int& refhundred { hundred }; // reference to int

    // trying to declare a reference to int&
    int&& refrefhundred {
        refhundred // compiler assumes int&& to be an rvalue reference and says that rvalue reference cannot bind to an lvalue
    };

    // let's try using auto
    auto& refref { refhundred }; // type is int& i.e this is another reference to hundred not a reference to reference

    // say hello to reference collapsing
    static_assert(std::is_same_v<::add_lvalue_reference_t<float>, float&>);
    static_assert(std::is_same_v<::add_lvalue_reference_t<float&>, float&>);
    static_assert(std::is_same_v<::add_lvalue_reference_t<float&&>, float&>);

    static_assert(std::is_same_v<::add_rvalue_reference_t<float>, float&&>);
    static_assert(std::is_same_v<::add_rvalue_reference_t<float&>, float&>);
    static_assert(std::is_same_v<::add_rvalue_reference_t<float&&>, float&&>);
}

/*
    THIS IS HOW REFERENCE COLLAPSING IN C++ WORKS

    ORIGINAL        ADDITION        RESULT
    T&              &               T&
    T&              &&              T&
    T&&             &               T&
    T&&             &&              T&&

*/
