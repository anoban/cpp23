#include <iostream>
#include <functional>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>

const size_t NELEMENTS = 1000;

void SayBye() throw(){
    ::_putws(L"GoodBye!");    
}


template<typename _Arg1Type, typename _Arg2Type, typename _ResultType> struct binary_function {
	typedef _Arg1Type   first_argument_type;
	typedef _Arg2Type   second_argument_type;
	typedef _ResultType result_type;
};

template<typename T> struct power {
    typedef T   first_argument_type;
	typedef T   second_argument_type;
	typedef T   result_type;

    result_type operator()(first_argument_type _arg1, second_argument_type _arg2) const throw() { 
        if(!_arg1) return 0;
        T result = _arg1;
        for(unsigned i = 1; i < _arg2; ++i) result *= _arg1;
        return result;
    }
};

int main(){
	std::srand(::time(NULL));

	std::wcout << L"Hi there! I'm using Microsoft Visual C++ 2008 Express" << std::endl;
	std::wcout << L"__cplusplus is " << __cplusplus << L" \n";

	std::vector<int> randoms(NELEMENTS);
	std::vector<int> _randoms(NELEMENTS);

	for(std::vector<int>::iterator it = randoms.begin(), end = randoms.end(); it != end; ++it) *it = rand() % 10;
	// std::transform(randoms.begin(), randoms.end(), _randoms.begin(), std::bind1st(std::multiplies<int>(), 2));
    std::transform(randoms.begin(), randoms.end(), _randoms.begin(), std::bind2nd(::power<int>(), 3));
	
	for(size_t i = 0; i < NELEMENTS; ++i){
		// DO NOT FORGET WE ARE USING A X86 COMPILER
		::wprintf_s(L"%4lu) %3d^3 = %3d\n", i, randoms.at(i), _randoms.at(i));
		// std::wcout << randoms.at(i) << L' ' << _randoms.at(i) << L'\n';	
	}
	
	::_putws(L"Done!");
	std::getchar();
    ::atexit(SayBye);
	return EXIT_SUCCESS;
}