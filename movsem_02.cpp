// g++ movsem_02.cpp -Wall -Wextra -O3 -Wpedantic -std=c++xx -o movsem_02.exe

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

// <cstdint> requires C++11 and later with g++ hence won't compile with
// -std=c++03

#if (__cplusplus >= 201103L)
    #define NOEXCEPT noexcept
#else
    #define NOEXCEPT throw()
#endif

class string {
    private:
        unsigned _size;
        unsigned _capacity; // at construction, capacity will be wtice the size of the
                            // string (including the null terminator)
        char*    _resource;

    public:
        typedef char*       pointer;
        typedef const char* const_pointer;
        typedef char*       iterator;
        typedef const char* const_iterator;

        inline string() NOEXCEPT : _size(0), _capacity(0), _resource(NULL) { }

        inline explicit string(const unsigned& size) NOEXCEPT :
            _size(size),
            _capacity(size * 2),
            _resource(new (std::nothrow) char[size * 2]) {
            if (!_resource) { // has the allocation failed,
                _capacity = _size = 0;
                ::fputws(L"allocation failure!\n", stderr);
            }
        }

        inline string(const char& ch, const unsigned& repeats) NOEXCEPT :
            _size(repeats),
            _capacity(repeats * 2),
            _resource(new (std::nothrow) char[_capacity]) {
            if (!_resource) { // has the allocation failed,
                _capacity = _size = 0;
                ::fputws(L"allocation failure!\n", stderr);
                return;
            }

            ::memset(_resource, ch, repeats);
            ::memset(_resource + repeats, 0U, _capacity - _size);
        }

        inline explicit string(const char *str) NOEXCEPT
      : // not a converting constructor
        _size(::strlen(str)),
        _capacity(_size * 2 + 1),
        _resource(new(std::nothrow) char[_capacity]) {
            if (!_resource) { // has the allocation failed,
                _capacity = _size = 0;
                ::fputws(L"allocation failure!\n", stderr);
            }
            ::memset(_resource, 0U, _capacity);
            ::strcpy_s(_resource, _capacity, str);
        }

        inline string(const string& other) NOEXCEPT :
            // copied from buffer may have trailing junk bytes, we do not want them
            // here, hence using ._size instead of _capacity
            _size(other._size),
            _capacity(other._size * 2 + 1),
            _resource(new (std::nothrow) char[_capacity]) {
            if (!_resource) { // has the allocation failed,
                _capacity = _size = 0;
                ::fputws(L"allocation failure!\n", stderr);
                return;
            }

            ::memcpy_s(_resource, _size, other._resource, other._size);
            ::_putws(L"copy construction!");
        }

        inline string& operator=(const string& other) NOEXCEPT {
            if (this == &other) return *this;

            if (_capacity == other._size) { // no need for new allocations
                ::memcpy_s(_resource, _capacity, other._resource, other._size);
                _size = _capacity = other._size;
                return *this;
            }

            if (_capacity > other._size) { // no need for new allocations
                ::memset(_resource, 0U,
                         _capacity); // since we'll have trailing garbage bytes
                ::memcpy_s(_resource, _size, other._resource, other._size);
                _size = other._size; // don't bother the _capacity
                return *this;
            }

            delete[] _resource;                               // give up the old buffer
            _resource = new (std::nothrow) char[other._size]; // allocate a new buffer

            if (!_resource) { // has the allocation failed,
                _size = _capacity = 0;
                ::fputws(L"allocation failure!\n", stderr);
                return *this;
            }

            _size = _capacity = other._size;
            ::memcpy_s(_resource, _size, other._resource, other._size);
            ::_putws(L"copy assignment!");
            return *this;
        }

#if (__cplusplus >= 201'103L) // implement move semantics

        inline string(string&& other) NOEXCEPT : _size(other._size), _capacity(other._capacity), _resource(other._resource) {
            other._resource = nullptr;
            other._size = other._capacity = 0;
        }

        inline string& operator=(string&& other) NOEXCEPT {
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

        inline size_t length() const NOEXCEPT { return _size; }

        inline ~string() NOEXCEPT { delete[] _resource; }

        inline char* c_str() NOEXCEPT { return _resource; }

        inline const char* c_str() const NOEXCEPT { return _resource; }

        string operator+(const string& other) const NOEXCEPT { }

        string operator+=(const string& other) const NOEXCEPT { }
};

static inline ::string skyfall() NOEXCEPT {
    return ::string("Skyfall is where we start\n"
                    "A thousand miles and poles apart\n"
                    "Where worlds collide and days are dark\n"
                    "You may have my number, you can take my name\n"
                    "But you'll never have my heart\n");
}

static const size_t MiB = 1'024 * 1'024;

int main() {
    const ::string empty; // default construction
    ::string       onemib(7 * MiB);
    const ::string jbond("I've drowned and dreamt this moment.... so overdue I "
                         "owe them................");
    ::puts(jbond.c_str());

    string adele;        // default construction
    adele = ::skyfall(); // copy assignment in C++03, move assignment in C++11 and
                         // later

    ::puts(adele.c_str());

    const string aaaaa('A', 50);
    ::puts(aaaaa.c_str());

    return EXIT_SUCCESS;
}
