#include <iostream>

template<class T> class Employee {
        void func() throw() { ::puts("Hi Employee!"); }
};

// specialization of the func method
template<> void Employee<double>::func() throw() { ::puts("Hello!"); }

// class template with two template parameters
template<class T, class U> class Manager {
        void func() throw();
};

// generic implementation
template<class T, class U> void Manager<T, U>::func() throw() { ::puts("Hi Manager!"); }

// partial specialization of func method for Manager class
// class methods does not allow partial specialization on class template arguments
// THIS IS NOT ABOUT TEMPLATED MEMBER FUNCTIONS
template<class T> void Manager<float>::func() throw() { ::puts("Partial specialization of member functions IS NOT ALLOWED!"); }

// complete specialization
template<> void Manager<int, float>::func() throw() { ::puts("Full specializations are okay!"); }

// policies are quite similar to traits (as in Rust impls) but they place a higher emphasis on behaviour instead of types
