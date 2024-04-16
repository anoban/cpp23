// g++ anglebrackets.cpp  -Wall -std=c++03 -O3 -Wextra -Wpedantic
// even if a templated class type has defaults for all parameters, the angle brackets cannot be omitted completely (in C++03)

template<typename T1 = int, typename T2 = double, typename T3 = short, typename T4 = float> class defaults { };

// templates simply serve as moulds to create new class types or functions
// type qualifiers will affect the type of the template instantiation
// e.g. sum<long>() and sum<const long>() are two different functions

template<typename _ArgType, typename _ResType, _ResType (*fnptr)(_ArgType)> double apply() noexcept {
    // third template parameter is a function pointer type with a signature of
    // _ResType func(_ArgType)
}

// template parameters downstream can depend on upstream template type or non-type arguments
template<typename T, const T value> struct some { };

// plain class types can have templated member functions
struct pod_struct {
        template<typename scalar_t> scalar_t abs(scalar_t value) noexcept { return value < 0 ? value * -1 : value; }
};

// templated class types can also have templated member functions
template<class T> class Class {
        template<class U> U function() throw() { }
};

int main() {
    defaults            object;
    some<long, 0.76576> error;
    some<float, 7567>   error_still;

    return 0;
}

// g++ with standards below -std=c++17 will give the following error
/*
anglebrackets.cpp:6:14: error: missing template arguments before 'object'
    6 |     defaults object;
      |              ^~~~~~
*/
