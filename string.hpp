#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

#include <sal.h>

namespace value_semantics {
    class string { // a super trivial string class with no non-trivial optimizations whatsoever
        private:
            unsigned _size;     // does not count the null terminator
            unsigned _capacity; // at construction, capacity will be twice the size of the string (including the null terminator)
            char*    _resource;

        public:
            inline string() throw() : _size(0), _capacity(0), _resource(NULL) { }

            inline string(_In_ const char& ch, _In_ const unsigned long& repeats) throw() :
                _size(repeats), _capacity(repeats * 2), _resource(new (std::nothrow) char[_capacity]) {
                if (!_resource) { // has the allocation failed,
                    _capacity = _size = 0;
                    ::fputws(L"allocation failure!\n", stderr);
                    return;
                }

                ::memset(_resource, 0U, _capacity);
                ::memset(_resource, ch, repeats);
            }

            inline explicit string(_In_ const char* const str) throw() : // not a converting constructor
                _size(::strlen(str)), _capacity(_size * 2 + 1), _resource(new (std::nothrow) char[_capacity]) {
                if (!_resource) { // has the allocation failed,
                    _capacity = _size = 0;
                    ::fputws(L"allocation failure!\n", stderr);
                    return;
                }

                ::memset(_resource, 0U, _capacity);
                ::strcpy_s(_resource, _size, str);
            }

            inline string(_In_ const string& other) throw() : // not exactly a trivial copy constructor
                _size(other._size), _capacity(other._size * 2 + 1), _resource(new (std::nothrow) char[_capacity]) {
                if (!_resource) { // has the allocation failed,
                    _capacity = _size = 0;
                    ::fputws(L"allocation failure!\n", stderr);
                    return;
                }

                ::memset(_resource, 0U, _capacity);
                ::strcpy_s(_resource, _size, other._resource);
                ::_putws(L"copy construction!");
            }

            inline string& operator=(_In_ const string& other) throw() { // not exactly a trivial copy assignment operator
                if (this == &other) return *this;

                delete[] _resource; // give up the old buffer
                _size     = other._size;
                _capacity = other._size * 2;
                _resource = new (std::nothrow) char[_capacity]; // allocate a new buffer

                if (!_resource) { // has the allocation failed,
                    _size = _capacity = 0;
                    ::fputws(L"allocation failure!\n", stderr);
                    return *this;
                }

                ::memset(_resource, 0U, _capacity);
                ::strcpy_s(_resource, _size, other._resource);
                ::_putws(L"copy assignment!");
                return *this;
            }

#if (__cplusplus >= 201103L) // implement move semantics

            inline string(_Inout_ string&& other) noexcept : _size(other._size), _capacity(other._capacity), _resource(other._resource) {
                other._resource = nullptr;
                other._size = other._capacity = 0;
                ::_putws(L"move construction!");
            }

            inline string& operator=(_Inout_ string&& other) noexcept {
                if (this == &other) return *this;

                delete[] _resource; // give up the old buffer
                _size           = other._size;
                _capacity       = other._capacity;
                _resource       = other._resource;

                other._resource = nullptr;
                other._size = other._capacity = 0;

                ::_putws(L"move assignment!");
                return *this;
            }

#endif

            static inline string with_capacity(_In_ const unsigned long& size) throw() {
                string temporary;
                temporary._resource = new (std::nothrow) char[size];
                if (!temporary._resource) { // has the allocation failed,
                    ::fputws(L"allocation failure!\n", stderr);
                    return temporary;
                }

                temporary._capacity = size;
                return temporary;
            }

            inline size_t length() const throw() { return _size; }

            inline ~string() throw() {
                delete[] _resource;
                _resource = NULL;
                _size = _capacity = 0;
            }

            inline char* c_str() throw() { return _resource; }

            inline const char* c_str() const throw() { return _resource; }

            string operator+(_In_ const string& other) const throw() { }

            string operator+=(_In_ const string& other) const throw() { }
    };

} // namespace value_semantics
