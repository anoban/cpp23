// clang .\cpp98lambdas.cpp -Wall -O3 -std=c++98 -Wextra -pedantic
// won't work as MSVC++'s STL doesn't support any standards prior to C++14 WOW!
// and clang relies on MSVC++'S STL headers.

#include <algorithm>
#include <iostream>
#include <vector>

// functors are very useful, but C++98/03 local class types didn't support templates
// who the fuck cares about C++98/03 anyways :(
// besides MSVC's STL seems to be fucked up when it comes to supporting older standards

int main() {
    struct functor { // local class type

            void operator()(int x) const { std::wcout << x << L' '; }
    };

    // with C++11 local class types can be declared with operator()
    // and C++11 brough lambdas to the table.

    const std::vector<int> coll(20, 7);                 // auto didn't work either
    std::for_each(coll.begin(), coll.end(), functor()); // .cebegin() and .cend() methods seem unavailable in C++98
    // in C++98 locally defined types cannot be used as template arguments!
    return EXIT_SUCCESS;
}

// in C++98 templates could not be instantiated with local types
// std::for_each is a templated algorithm, so won't work with local classes in C++98

/*
$ g++ cpp98lambdas.cpp -Wall -O3 -std=c++98 gives the following diagnostics

cpp98lambdas.cpp: In function 'int main()':
cpp98lambdas.cpp:19:18: error: no matching function for call to 'for_each(std::vector<int>::const_iterator, std::vector<int>::const_iterator, main()::functor)'
   19 |     std::for_each(coll.begin(), coll.end(), functor()); // .cebegin() and .cend() methods seems unavailable in C++98
      |     ~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In file included from C:/msys64/ucrt64/include/c++/13.2.0/algorithm:61,
                 from cpp98lambdas.cpp:5:
C:/msys64/ucrt64/include/c++/13.2.0/bits/stl_algo.h:3827:5: note: candidate: 'template<class _IIter, class _Funct> _Funct std::for_each(_IIter, _IIter, _Funct)'
 3827 |     for_each(_InputIterator __first, _InputIterator __last, _Function __f)
      |     ^~~~~~~~
C:/msys64/ucrt64/include/c++/13.2.0/bits/stl_algo.h:3827:5: note:   template argument deduction/substitution failed:
cpp98lambdas.cpp: In substitution of 'template<class _IIter, class _Funct> _Funct std::for_each(_IIter, _IIter, _Funct) [with _IIter = __gnu_cxx::__normal_iterator<const int*, std::vector<int> >; _Funct = main()::functor]':
cpp98lambdas.cpp:19:18:   required from here
cpp98lambdas.cpp:19:18: error: template argument for 'template<class _IIter, class _Funct> _Funct std::for_each(_IIter, _IIter, _Funct)' uses local type 'main()::functor'
   19 |     std::for_each(coll.begin(), coll.end(), functor()); // .cebegin() and .cend() methods seems unavailable in C++98
      |     ~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cpp98lambdas.cpp:19:18: error:   trying to instantiate 'template<class _IIter, class _Funct> _Funct std::for_each(_IIter, _IIter, _Funct)'

*/
