#define __SSTRING_PRINT_METHOD_SIGNATURES_ON_CALL__
#include <sstring>
#include <vector>

// clang-format off

int main() {
	std::vector< ::sstring> names;
	::sstring Julia ("Julia");
	const ::sstring Natalie ("Natalie");
	
	names.push_back("James");
	names.push_back(Julia);
	names.push_back(Natalie);
	names.push_back(Julia + " Devries");
	
	for(std::vector< ::sstring>::iterator it = names.begin(), end = names.end(); it != end; ++it){
		::puts(it->c_str());
	}
	return EXIT_SUCCESS;
}

// clang-format on