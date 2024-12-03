template<typename T> struct test {
        typedef float f32; // f32 is a dependent name as its meaning depends on T
        typedef T     value_type;
};

// here's a specialization for int
template<> struct test<int> {
        // as long as the template argument is not int, f32 will just be a type alias to float
        static const long f32 = 436254; // in the main template f32 was defined as a typedef to float!
        typedef int       value_type;
};

struct plain {
        typedef short u16;
        using i32 = int;
        static const float pi;
};

template<typename T> struct testest {
        // test<T>::value_type member_0; //  error: need 'typename' before 'test<T>::value_type' because 'test<T>' is a dependent scope
        typename test<T>::value_type member_1;
        // test<T>::f32                 whatnow; // error: need 'typename' before 'test<T>::f32' because 'test<T>' is a dependent scope
        typename test<T>::f32        how_about_now;
};

int main() {
    const int          num = 654;
    const float        pi  = 3.1415927F;
    test<wchar_t>::f32 x; // declaration of variable x of type float T != int
    static_assert(
        test<int>::f32 * num == 436254 * num, L"Must be equal!"
    ); // a multiplication expression, test<int>::f32 evaluates to an integer literal

    const test<float>::f32* const        atpi  = &pi; // variable definition, test<float>::f32 evaluates to float
    const test<float>::value_type* const atpi2 = &pi; // same thing

    plain::i32                           value = num; // WOW
    // using the scope resolution operator :: to access typedefs works for plain non-templated class types too.

    testest<int>::how_about_now; //  error: 'typename test<int>::f32' names 'const long int test<int>::f32', which is not a type
    testest<long>::how_about_now;

    return 0;
}
